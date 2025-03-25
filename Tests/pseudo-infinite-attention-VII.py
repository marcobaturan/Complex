# =============================================================================
# IMPORT MODULES WITH ERROR HANDLING
# =============================================================================
import psutil

try:
    import os
    import sqlite3
    import hashlib
    import threading
    import pandas as pd
    from functools import lru_cache
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from llama_cpp import Llama
    import time
    from sentence_transformers import SentenceTransformer, util
    print("All modules loaded.")
except Exception as e:
    print("Some modules are missing: " + str(e))

# =============================================================================
# GLOBAL CONFIGURATIONS
# =============================================================================

SQLITE_DB_PATH = "infinite_attention_cache.db"
MAX_WORKERS = 8
LRU_CACHE_SIZE = 128
CHUNK_SIZE = 512
MODEL_PATH = "../Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
model_embedding = SentenceTransformer('all-MiniLM-L6-v2')

# =============================================================================
# VERIFY MODEL LOADING
# =============================================================================
try:
    model = Llama(model_path=MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
# =============================================================================
# INITIALIZE SQLITE CACHE
# =============================================================================

def init_sqlite_cache(db_path: str):
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

global_model = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,       # Reduce the context for lower RAM consumption
    n_batch=256,      # Reduce batch
    max_tokens=400,   # Reduce output
    verbose=False,
    stop=["PASS", "\n"],
    echo=False
)

# =============================================================================
# MODEL CALL WITH MEMOIZATION
# =============================================================================

@lru_cache(maxsize=LRU_CACHE_SIZE)
def call_tinyllama(prompt: str) -> str:
    result = global_model(prompt)['choices'][0]['text'].strip()
    if not result:  # if get void avoid clash
        return "PASS"
    return result

# =============================================================================
# SQLITE CACHE FUNCTIONS
# =============================================================================

def generate_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_cached_result(chunk: str) -> str:
    key = generate_hash(chunk)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT result FROM cache WHERE key=?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def set_cached_result(chunk: str, result: str):
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

def process_fragment(fragment: str, user_message: str) -> str:
    # the fragment has more than 900 tokens, shorten it
    if len(fragment.split()) > 900:
        fragment = " ".join(fragment.split()[:900])  # shorten to 900 tokens
    prompt = (
        "You are reconstructing knowledge from different text fragments. "
        "Your goal is to maintain internal consistency and logical accuracy. "
        "If the provided fragment does not contain the answer, return 'PASS'. "
        f"Context: {fragment} | User: {user_message}"
    )

    result = call_tinyllama(prompt)
    return result if 'PASS' not in result else 'PASS'

model_lock = threading.Lock()  # Lock global para evitar concurrencia

def process_chunk(chunk: str, user_message: str) -> str:
    cached = get_cached_result(chunk)
    if cached is not None:
        return cached
    with model_lock:  # Only one call per thread to avoid model crash
        result = process_fragment(chunk, user_message)
    set_cached_result(chunk, result)
    return result

def filter_results(results: list, exclude='PASS') -> list:
    return [result for result in results if result != exclude]

# =============================================================================
# SPLITTING THE MINDFILE INTO CHUNKS
# =============================================================================

def split_mindfile(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# =============================================================================
# AGGREGATION OF CHUNK RESPONSES
# =============================================================================

def aggregate_responses(responses: list, user_message: str) -> str:
    embeddings = model_embedding.encode([user_message] + responses)
    user_embed = embeddings[0]
    scores = [util.cos_sim(user_embed, emb).item() for emb in embeddings[1:]]

    best_idx = scores.index(max(scores))
    return responses[best_idx]  # Seleccionar la respuesta más coherente

# =============================================================================
# MAIN FUNCTION: INFINITE ATTENTION PIPELINE
# =============================================================================

def infinite_attention_pipeline(mindfile_text: str, user_message: str) -> str:
    chunks = split_mindfile(mindfile_text)
    responses = []
    processing_details = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {executor.submit(process_chunk, chunk, user_message): chunk for chunk in chunks}
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                response = future.result()
                processing_details.append({'Fragment': chunk[:50], 'Response': response})
                responses.append(response)
            except Exception as e:
                processing_details.append({'Fragment': chunk[:50], 'Response': f'Error: {e}'})

    valid_results = filter_results(responses)
    aggregated_output = aggregate_responses(valid_results, user_message)
    final_output = call_tinyllama(aggregated_output)

    details_df = pd.DataFrame(processing_details)
    print("\nProcessing Details:")
    print(details_df)
    return final_output

# =============================================================================
# INTERACTIVE LOOP WITH EXIT CONDITION
# =============================================================================

def interactive_chat():
    if os.path.exists("MindFile.txt"):
        with open("MindFile.txt", "r", encoding="utf-8") as f:
            mindfile_text = f.read()
    else:
        print("MindFile.txt not found. Using default test string.")


    while True:
        start = time.time()
        # start monitoring
        process = psutil.Process(os.getpid())
        user_message = input("User: ")
        if user_message.lower() in ["exit", "bye"]:
            print("System: Goodbye!")
            break
        response = infinite_attention_pipeline(mindfile_text, user_message)
        print("System:", response)
        end = time.time()
        print(
            f"⏳ Cognitive cost: Total time: {end-start:.2f} s | 🔹 CPU: {psutil.cpu_percent()}% | RAM: {process.memory_info().rss / (1024 * 1024)} MB")


if __name__ == "__main__":
    start = time.time()
    # Monitoreo inicial
    process = psutil.Process(os.getpid())
    print(f"🔹 CPU: {psutil.cpu_percent()}% | RAM: {process.memory_info().rss / (1024 * 1024)} MB")
    interactive_chat()
    end = time.time()
    print(
        f"⏳ Accumulated cognitive cost: Global time: {end - start:.2f} s | 🔹 CPU: {psutil.cpu_percent()}% | RAM: {process.memory_info().rss / (1024 * 1024)} MB")
