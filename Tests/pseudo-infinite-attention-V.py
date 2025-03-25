import sqlite3
import hashlib
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from llama_cpp import Llama
import pandas as pd

# =============================================================================
# GLOBAL CONFIGURATIONS
# =============================================================================

SQLITE_DB_PATH = "infinite_attention_cache.db"
MAX_WORKERS = 8
LRU_CACHE_SIZE = 128
CHUNK_SIZE = 1000

# =============================================================================
# INITIALIZE SQLITE CACHE
# =============================================================================

def init_sqlite_cache(db_path: str):
    """Initialize the SQLite database to store chunk results."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            result TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_sqlite_cache(SQLITE_DB_PATH)
sql_lock = threading.Lock()

# =============================================================================
# GLOBAL MODEL INITIALIZATION
# =============================================================================

# Initialize the Llama model once, instead of every time in call_tinyllama
global_model = Llama(
    model_path="../Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=2048,       # Context window size
    n_batch=512,      # Batch processing size
    verbose=False,    # Disable verbose logging
    max_tokens=1200,  # Limit response length
    stop=["PASS", "\n"],  # Stop generation conditions
    echo=False       # Do not echo the prompt
)

# =============================================================================
# MODEL CALL WITH MEMOIZATION
# =============================================================================

@lru_cache(maxsize=LRU_CACHE_SIZE)
def call_tinyllama(prompt: str) -> str:
    """
    Call the quantized TinyLlama model with the given prompt.
    Uses a globally initialized model for efficiency.
    """
    # For debugging: print a snippet of the prompt (first 50 chars)
    print(f"call_tinyllama: Processing prompt (first 50 chars): {prompt[:50]}...")
    result = global_model(prompt)['choices'][0]['text'].strip()
    # For debugging: print a snippet of the result
    print(f"call_tinyllama: Model output (first 50 chars): {result[:50]}...")
    return result

# =============================================================================
# SQLITE CACHE FUNCTIONS
# =============================================================================

def generate_hash(text: str) -> str:
    """Generate a SHA256 hash for the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_cached_result(chunk: str) -> str:
    """Retrieve cached result for a chunk, if available."""
    key = generate_hash(chunk)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT result FROM cache WHERE key=?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def set_cached_result(chunk: str, result: str):
    """Store the processed result of a chunk into the SQLite cache."""
    key = generate_hash(chunk)
    with sql_lock:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO cache (key, result) VALUES (?, ?)", (key, result))
        conn.commit()
        conn.close()

# =============================================================================
# CHUNK PROCESSING FUNCTION
# =============================================================================

def process_chunk(chunk: str) -> str:
    """
    Process a single chunk:
      1. Check SQLite cache.
      2. Call the TinyLlama model if not cached.
      3. Cache and return the result.
    """
    cached = get_cached_result(chunk)
    if cached is not None:
        print("process_chunk: Cache hit for a chunk.")
        return cached

    print("process_chunk: Processing new chunk.")
    result = call_tinyllama(chunk)
    set_cached_result(chunk, result)
    return result

# =============================================================================
# SPLITTING THE MINDFILE INTO CHUNKS
# =============================================================================

def split_mindfile(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Split the mindfile text into chunks. Splitting is done by whitespace.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    print(f"split_mindfile: Generated {len(chunks)} chunks.")
    return chunks

# =============================================================================
# AGGREGATION OF CHUNK RESPONSES
# =============================================================================

def aggregate_responses(responses: list) -> str:
    """
    Aggregate individual chunk responses into a final output.
    """
    aggregated = "\n".join(responses)
    print("aggregate_responses: Aggregated output length:", len(aggregated))
    return aggregated

# =============================================================================
# MAIN FUNCTION: INFINITE ATTENTION PIPELINE
# =============================================================================

def infinite_attention_pipeline(mindfile_text: str) -> str:
    """
    Process the mindfile by splitting it into chunks, processing in parallel,
    aggregating results, and then refining the final output.
    """
    if not mindfile_text.strip():
        print("infinite_attention_pipeline: Empty mindfile input!")
        return "No input data provided."

    chunks = split_mindfile(mindfile_text)
    responses = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {executor.submit(process_chunk, chunk): idx for idx, chunk in enumerate(chunks)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                responses[idx] = future.result()
                print(f"infinite_attention_pipeline: Chunk {idx} processed.")
            except Exception as e:
                responses[idx] = f"Error processing chunk {idx}: {e}"
                print(f"infinite_attention_pipeline: Error processing chunk {idx}: {e}")

    aggregated_output = aggregate_responses(responses)
    # Optionally refine the aggregated result with another model call
    final_output = call_tinyllama(aggregated_output)
    return final_output

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Check if MindFile.txt exists; if not, set a default mindfile_text.
    if os.path.exists("MindFile.txt"):
        with open("MindFile.txt", "r", encoding="utf-8") as f:
            mindfile_text = f.read()
    else:
        print("MindFile.txt not found. Using default test string.")
        mindfile_text = "This is a default mindfile text. " * 1000  # Default fallback

    final_result = infinite_attention_pipeline(mindfile_text)
    print("Final Aggregated Result:")
    print(final_result)
