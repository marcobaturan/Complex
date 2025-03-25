# =============================================================================
# IMPORT MODULES WITH ERROR HANDLING
# =============================================================================

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
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import psutil
    import random
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
#MODEL_PATH = "../Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = "../Models/gemma-3-1b-it-Q8_0.gguf"
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
# INITIALIZE SQLITE CACHE AND CLEAR CACHE
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


def clear_sqlite_cache(db_path: str):
    """
    Clear all entries from the SQLite cache database.

    Args:
        db_path (str): Path to the SQLite database
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cache")
    conn.commit()
    conn.close()
    print("SQLite cache cleared.")

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
# RANKING IMPORTANT TERMS
# =============================================================================

def extract_important_terms(text, num_terms=10):
    """
    Automatically extract the most important terms of Mindfile using TF-IDF.
    """
    sentences = text.split(". ")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]

    if len(tfidf_sorting) == 0:
        return []

    top_terms = feature_array[tfidf_sorting][:num_terms]
    return list(top_terms)


# =============================================================================
# GOUPING SEMANTIC TERMS
# ============================================================================


def find_related_terms(text, important_terms):
    """
    Group terms semantically related to the key terms using embeddings.
    """
    sentences = text.split(". ")
    if not sentences:
        return {}

    sentence_embeddings = model_embedding.encode(sentences)
    related_terms = {}

    for term in important_terms:
        term_embedding = model_embedding.encode(term)
        scores = [util.cos_sim(term_embedding, sent_emb).item() for sent_emb in sentence_embeddings]

        if scores:
            best_match_idx = np.argmax(scores)
            related_terms[term] = sentences[best_match_idx]

    return related_terms

# =============================================================================
# VALIDATION ANSWERS
# =============================================================================

def validate_response(response, extracted_facts):
    """
    Verify if the answer contains information consistent with the terms extracted.
    """
    if not extracted_facts:
        return response

    if not any(term in response for term in extracted_facts.keys()):
        best_fact = max(extracted_facts.keys(), key=lambda t: util.cos_sim(
            model_embedding.encode(t), model_embedding.encode(response)).item()
                        )
        return f"I'm not sure, but {extracted_facts[best_fact]}"

    return response
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

def process_fragment(fragment: str, user_message: str, extracted_facts: dict[str, str]) -> str:
    # the fragment has more than 900 tokens, shorten it
    if len(fragment.split()) > 900:
        fragment = " ".join(fragment.split()[:900])  # shorten to 900 tokens
    identity_context = " ".join(extracted_facts.values())
    prompt = (
        "You are reconstructing knowledge from different text fragments. "
        "Your goal is to maintain internal consistency and logical accuracy. "
        "Here are key facts: " f"{identity_context} "
        "If the provided fragment does not contain the answer, return 'PASS'. "
        "Now, given the following passage, answer while keeping consistency: "
        f"Context: {fragment} | User: {user_message}"
    )

    result = call_tinyllama(prompt)
    return validate_response(result, extracted_facts) if 'PASS' not in result else 'PASS'

model_lock = threading.Lock()  # Lock global para evitar concurrencia


def process_chunk(chunk: str, user_message: str, extracted_facts: dict) -> str:
    cached = get_cached_result(chunk)
    if cached is not None:
        return cached

    with model_lock:  # Solo una llamada al modelo por hilo para evitar crashes
        result = process_fragment(chunk, user_message, extracted_facts)

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
    if not responses:
        return "I'm not sure, I need more information."

    embeddings = model_embedding.encode([user_message] + responses)
    user_embed = embeddings[0]
    scores = [util.cos_sim(user_embed, emb).item() for emb in embeddings[1:]]

    if not scores:
        return "I'm not sure, but I need more details."

    best_idx = scores.index(max(scores))
    return responses[best_idx]


# =============================================================================
# MAIN FUNCTION: INFINITE ATTENTION PIPELINE
# =============================================================================
# Before the infinite_attention_pipeline function, add this function
# Before the infinite_attention_pipeline function, add this function
def extract_facts_from_mindfile(mindfile_text: str):
    """
    Extract important terms and their context from the mindfile.

    Args:
        mindfile_text (str): The entire text of the mindfile

    Returns:
        dict: A dictionary of important terms and their related sentences
    """
    important_terms = extract_important_terms(mindfile_text)
    related_terms = find_related_terms(mindfile_text, important_terms)
    return related_terms


def infinite_attention_pipeline(mindfile_text: str, user_message: str) -> str:
    # First, extract facts from the entire mindfile
    extracted_facts = extract_facts_from_mindfile(mindfile_text)

    chunks = split_mindfile(mindfile_text)
    responses = []
    processing_details = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Pass extracted_facts to process_chunk
        future_to_chunk = {executor.submit(process_chunk, chunk, user_message, extracted_facts): chunk for chunk in
                           chunks}
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
def reset_lru_cache():
    """
    Reset the LRU cache for model calls.
    """
    call_tinyllama.cache_clear()
    print("LRU cache cleared.")

def interactive_chat():
    if os.path.exists("MindFile.txt"):
        with open("MindFile.txt", "r", encoding="utf-8") as f:
            mindfile_text = f.read()
    else:
        print("MindFile.txt not found. Using default test string.")

    while True:
        # Clear caches before each interaction
        clear_sqlite_cache(SQLITE_DB_PATH)
        reset_lru_cache()

        start = time.time()
        process = psutil.Process(os.getpid())
        user_message = input("User: ")

        if user_message.lower() in ["exit", "bye"]:
            print("System: Goodbye!")
            break

        response = infinite_attention_pipeline(mindfile_text, user_message)
        print("System:", response)

        end = time.time()
        print(
            f"⏳ Cognitive cost: Total time: {end - start:.2f} s | 🔹 CPU: {psutil.cpu_percent()}% | RAM: {process.memory_info().rss / (1024 * 1024)} MB")


if __name__ == "__main__":
    start = time.time()
    # Monitoreo inicial
    process = psutil.Process(os.getpid())
    print(f"🔹 CPU: {psutil.cpu_percent()}% | RAM: {process.memory_info().rss / (1024 * 1024)} MB")
    interactive_chat()
    end = time.time()
    print(
        f"⏳ Accumulated cognitive cost: Global time: {end - start:.2f} s | 🔹 CPU: {psutil.cpu_percent()}% | RAM: {process.memory_info().rss / (1024 * 1024)} MB")
