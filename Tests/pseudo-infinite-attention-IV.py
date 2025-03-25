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

# Path to the SQLite cache database for storing processed chunk results
SQLITE_DB_PATH = "infinite_attention_cache.db"

# Maximum number of threads to match the 8-core CPU
MAX_WORKERS = 8

# Maximum size for the LRU cache for model outputs (tune based on available memory)
LRU_CACHE_SIZE = 128

# Chunk size in tokens/words for splitting the mindfile (this is an approximate measure)
CHUNK_SIZE = 1000


# =============================================================================
# INITIALIZE SQLITE CACHE
# =============================================================================

def init_sqlite_cache(db_path: str):
    """
    Initialize the SQLite database to store chunk results.
    The table 'cache' will hold a key (hash of the chunk) and the corresponding model output.
    """
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


# Call initialization at startup
init_sqlite_cache(SQLITE_DB_PATH)

# Thread lock to protect SQLite writes from concurrent threads
sql_lock = threading.Lock()


# =============================================================================
# MODEL CALL WITH MEMOIZATION
# =============================================================================

@lru_cache(maxsize=LRU_CACHE_SIZE)
def call_tinyllama(prompt: str) -> str:
    """
    Call the quantized TinyLlama model with the given prompt.
    This function is decorated with lru_cache to memoize results for identical prompts.

    In a production system, replace the content of this function with the actual call
    to the TinyLlama model API or inference engine.
    """
    # Simulated model processing (replace with your model call)
    # Here, we simply return the prompt reversed as a dummy "response"
    # In practice, this will be your model's generated output.
    model = Llama(
        model_path="../Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        n_ctx=2048,  # Set context window size, give margin to max tokens
        n_batch=512,  # Set batch processing size
        verbose=False,  # Disable verbose logging
        max_tokens = 1200,  # Limit response length, give margin to chunk size
        stop=["PASS", "\n"],  # Stop generation conditions
        echo=False,  # Don't echo the prompt
    )
    result = model(prompt)['choices'][0]['text'].strip()
    return result


# =============================================================================
# SQLITE CACHE FUNCTIONS
# =============================================================================

def generate_hash(text: str) -> str:
    """
    Generate a SHA256 hash for the given text. This is used as a unique key
    for storing and retrieving processed chunks from the SQLite cache.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_cached_result(chunk: str) -> str:
    """
    Check the SQLite cache for a result corresponding to the given chunk.
    Return the cached result if found; otherwise, return None.
    """
    key = generate_hash(chunk)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT result FROM cache WHERE key=?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def set_cached_result(chunk: str, result: str):
    """
    Store the processed result of a chunk into the SQLite cache.
    Uses a thread lock to ensure database write consistency.
    """
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
    Process a single chunk of the mindfile:
      1. Check the SQLite cache for a precomputed result.
      2. If not cached, call the TinyLlama model to generate a response.
      3. Cache the result in SQLite for future reuse.

    The function returns the model output for the given chunk.
    """
    # Step 1: Check SQLite cache
    cached = get_cached_result(chunk)
    if cached is not None:
        return cached

    # Step 2: Process the chunk using TinyLlama (with LRU memoization applied)
    result = call_tinyllama(chunk)

    # Step 3: Store the result in SQLite cache
    set_cached_result(chunk, result)

    return result


# =============================================================================
# SPLITTING THE MINDFILE INTO CHUNKS
# =============================================================================

def split_mindfile(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Split the provided mindfile text into smaller chunks.
    For simplicity, this function splits by whitespace tokens.
    More advanced splitting (maintaining sentence boundaries) can be implemented.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        # Join words back into a chunk string
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# =============================================================================
# AGGREGATION OF CHUNK RESPONSES
# =============================================================================

def aggregate_responses(responses: list) -> str:
    """
    Aggregate the responses from individual chunks into a final output.
    The aggregation strategy is to simply concatenate the responses,
    but advanced techniques (e.g., weighted merging or further summarization)
    can be applied.
    """
    # For demonstration, join responses with newlines.
    # In practice, you might call the model once more to summarize or synthesize.
    return "\n".join(responses)


# =============================================================================
# MAIN FUNCTION TO PROCESS THE MINDFILE AND APPLY INFINITE ATTENTION
# =============================================================================

def infinite_attention_pipeline(mindfile_text: str) -> str:
    """
    Main pipeline that implements the infinite attention algorithm:
      1. Splits the mindfile into N chunks.
      2. Processes each chunk in parallel, using memoization and caching.
      3. Aggregates the partial results into a final response.

    The function returns the final aggregated response.
    """
    # Step 1: Split the mindfile into chunks
    chunks = split_mindfile(mindfile_text)

    # Container for processed responses
    responses = [None] * len(chunks)

    # Step 2: Process chunks in parallel using a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Dictionary to hold futures for tracking chunk processing
        future_to_index = {executor.submit(process_chunk, chunk): idx for idx, chunk in enumerate(chunks)}

        # Process each completed future
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                responses[idx] = future.result()
            except Exception as e:
                responses[idx] = f"Error processing chunk {idx}: {e}"

    # Step 3: Aggregate responses from each chunk
    aggregated_output = aggregate_responses(responses)

    # Optional: Further process the aggregated output with the model for a refined final answer.
    final_output = call_tinyllama(aggregated_output)

    return final_output


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Load the mindfile from a file (for demo purposes, we use a placeholder string)
    # In practice, replace the below line with code to read your mindfile from disk.
    if os.path.exists("MindFile.txt"):
        with open("MindFile.txt", "r", encoding="utf-8") as f:
            mindfile_text = f.read()
    else:
        print("MindFile.txt not found.")

    # Run the infinite attention pipeline on the mindfile
    final_result = infinite_attention_pipeline(mindfile_text)

    # Output the final result
    print("Final Aggregated Result:")
    print(final_result)
