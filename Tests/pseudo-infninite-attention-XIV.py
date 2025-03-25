try:
    import os
    import sys
    import sqlite3
    import hashlib
    import logging
    import time
    from typing import Optional, Dict, List
    import numpy as np
    import psutil
    from sentence_transformers import SentenceTransformer, util
    from sklearn.feature_extraction.text import TfidfVectorizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llama_cpp import Llama
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import pandas as pd  # Import pandas
    print("All modules loaded.")
except ImportError as e:
    missing_module = e.name
    print(f"Error: Missing module '{missing_module}'. Please install it using pip:")
    print(f"pip install {missing_module}")
    sys.exit(1)  # Exit the program as it cannot run without these modules
except Exception as e:
    print("An unexpected error occurred during import:", e)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('infinite_attention.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global list to store database interactions
db_interaction_log = []

class ModelLoader:
    """
    Handles loading language models, supporting both Hugging Face Transformers
    and local GGUF models.
    """

    @staticmethod
    def load_model(
        model_path: Optional[str] = None,
        model_name: Optional[str] = "../Models/gemma-3-1b-it-Q8_0.gguf",
        use_local: bool = False
    ):
        """
        Loads the specified language model.

        Args:
            model_path (str, optional): Path to a local model file (e.g., GGUF).
            model_name (str, optional): Name of a Hugging Face Transformers model.
            use_local (bool):  If True, load a local model from `model_path`.

        Returns:
            Tuple[Optional[AutoTokenizer], Optional[AutoModelForCausalLM | Llama]]:
            A tuple containing the tokenizer and the model. The tokenizer may be
            None if a GGUF model is loaded.

        Raises:
            Exception: If the model fails to load.
        """
        if use_local and model_path and model_path.endswith('.gguf'):
            # Load GGUF model using llama-cpp
            try:
                llama_model = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_batch=256,
                    max_tokens=400,
                    verbose=False
                )
                return None, llama_model  # No tokenizer for GGUF
            except Exception as e:
                logger.error(f"GGUF model loading failed: {e}")
                raise

        # Load Hugging Face Transformers model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            return tokenizer, model
        except Exception as e:
            logger.error(f"Hugging Face model loading failed: {e}")
            raise


class SemanticContextEngine:
    """
    Extracts semantic context from text using sentence transformers and TF-IDF.
    """

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the SemanticContextEngine.

        Args:
            embedding_model (str, optional):  Name of the sentence transformer model.

        Raises:
            Exception: If the embedding model fails to load.
        """
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            logger.error(f"Embedding model loading failed: {e}")
            raise

    def extract_important_terms(self, text: str, num_terms: int = 10) -> List[str]:
        """
        Extracts the most important terms from the given text using TF-IDF.

        Args:
            text (str): The input text.
            num_terms (int, optional): The number of top terms to extract.

        Returns:
            List[str]: A list of the most important terms.
        """
        try:
            sentences = text.split(". ")
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(sentences)
            feature_array = np.array(vectorizer.get_feature_names_out())
            tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
            return list(feature_array[tfidf_sorting][:num_terms]) if len(tfidf_sorting) > 0 else []
        except Exception as e:
            logger.error(f"Term extraction error: {e}")
            return []

    def find_related_terms(self, text: str, important_terms: List[str]) -> Dict[str, str]:
        """
        Finds sentences that provide context for the given important terms.

        Args:
            text (str): The input text.
            important_terms (List[str]): A list of important terms.

        Returns:
            Dict[str, str]: A dictionary where keys are terms and values are
            contextual sentences.
        """
        try:
            sentences = text.split(". ")
            if not sentences:
                return {}

            sentence_embeddings = self.embedding_model.encode(sentences)
            related_terms = {}

            for term in important_terms:
                term_embedding = self.embedding_model.encode(term)
                scores = [util.cos_sim(term_embedding, sent_emb).item() for sent_emb in sentence_embeddings]
                if scores:
                    best_match_idx = np.argmax(scores)
                    related_terms[term] = sentences[best_match_idx]
            return related_terms
        except Exception as e:
            logger.error(f"Related terms extraction error: {e}")
            return {}

def log_db_interaction(operation: str, query: str, response: Optional[str] = None, cache_hit: bool = False):
    """
    Logs database interactions for real-time tracing.

    Args:
        operation (str): The database operation performed (e.g., "GET", "CACHE").
        query (str): The query used for the operation.
        response (Optional[str]): The response received from the database (if applicable).
        cache_hit (bool): True if it was a cache hit, False otherwise.
    """
    db_interaction_log.append({
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Operation": operation,
        "Query": query,
        "Response": response if response is not None else "N/A",
        "Cache Hit": cache_hit
    })

class ResponseCache:
    """
    Caches query-response pairs to improve efficiency. Uses an SQLite database
    for storage.
    """

    def __init__(self, cache_db_path: str = "infinite_attention_cache.db"):
        """
        Initializes the ResponseCache.

        Args:
            cache_db_path (str, optional): Path to the SQLite database file.

        Raises:
            sqlite3.Error: If there is an error initializing the database.
        """
        self.cache_db_path = cache_db_path
        self._init_cache_database()

    def _init_cache_database(self):
        """
        Initializes the SQLite database with a table for storing query-response pairs.
        """
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS response_cache (
                        query_hash TEXT PRIMARY KEY,
                        response TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Cache database initialization error: {e}")
            raise

    def get_cached_response(self, query: str) -> Optional[str]:
        """
        Retrieves a cached response for the given query, if it exists.

        Args:
            query (str): The user's query.

        Returns:
            Optional[str]: The cached response, or None if not found.
        """
        log_db_interaction("GET", query)
        try:
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT response FROM response_cache WHERE query_hash = ?",
                    (query_hash,)
                )
                result = cursor.fetchone()
                if result:
                    log_db_interaction("GET", query, result[0], cache_hit=True)
                    return result[0]
                else:
                    log_db_interaction("GET", query, None, cache_hit=False)
                    return None
        except sqlite3.Error as e:
            logger.error(f"Cache retrieval error: {e}")
            return None

    def cache_response(self, query: str, response: str):
        """
        Caches the given query-response pair.

        Args:
            query (str): The user's query.
            response (str): The generated response.
        """
        log_db_interaction("CACHE", query, response)
        try:
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO response_cache (query_hash, response) VALUES (?, ?)",
                    (query_hash, response)
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Cache storage error: {e}")

    def clear_cache(self):
        """
        Clears the entire response cache.
        """
        log_db_interaction("CLEAR", "N/A")
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM response_cache")
                conn.commit()
            logger.info("Response cache cleared successfully.")
        except sqlite3.Error as e:
            logger.error(f"Cache clearing error: {e}")


class InfiniteAttentionEngine:
    """
    The main class that orchestrates the entire process of loading the MindFile,
    processing user input, generating responses, and caching.
    """

    def __init__(
        self,
        mindfile_path: str = "MindFile.txt",
        model_path: Optional[str] = "../Models/gemma-3-1b-it-Q8_0.gguf",
        model_name: str = "microsoft/DialoGPT-medium",
        embedding_model: str = 'all-MiniLM-L6-v2',
        cache_db_path: str = "infinite_attention_cache.db",
        chunk_size: int = 512
    ):
        """
        Initializes the InfiniteAttentionEngine.

        Args:
            mindfile_path (str, optional): Path to the MindFile.
            model_path (str, optional): Path to a local language model.
            model_name (str, optional): Name of a Hugging Face Transformers model.
            embedding_model (str, optional): Sentence transformer model name.
            cache_db_path (str, optional): Path to the cache database.
            chunk_size (int, optional): Size of text chunks for processing.

        Raises:
            Exception: If model loading or engine initialization fails.
        """
        self.semantic_engine = SemanticContextEngine(embedding_model)
        self.response_cache = ResponseCache(cache_db_path)

        self.mindfile_path = mindfile_path
        self.mindfile_text = self._load_mindfile()
        self.persona_profile = self._extract_persona_profile(self.mindfile_text)

        use_local = model_path and os.path.exists(model_path) and model_path.endswith('.gguf')

        try:
            self.tokenizer, self.text_model = ModelLoader.load_model(
                model_path=model_path,
                model_name=model_name,
                use_local=use_local
            )
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

        if isinstance(self.text_model, Llama):
            self._generate_response = self._generate_response_llama

        self.chunk_size = chunk_size

    def _load_mindfile(self) -> str:
        """
        Loads the content of the MindFile.

        Returns:
            str: The content of the MindFile as a string.

        Raises:
            FileNotFoundError: If the MindFile is not found.
        """
        try:
            with open(self.mindfile_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("MindFile not found. Using empty string.")
            return ""

    def _extract_persona_profile(self, mindfile_text: str) -> Dict[str, str]:
        """
        Extracts a detailed persona profile from the MindFile text.
        This includes key facts, personality traits, communication style, etc.

        Args:
            mindfile_text (str): The text content of the MindFile.

        Returns:
            Dict[str, str]: A dictionary representing the persona profile.
        """
        #  This is where the core logic for persona extraction resides.
        #  The current implementation is basic and needs to be replaced
        #  with more sophisticated NLP techniques.

        #  For now, let's extract important terms and related sentences as a starting point
        important_terms = self.semantic_engine.extract_important_terms(mindfile_text)
        persona_profile = self.semantic_engine.find_related_terms(mindfile_text, important_terms)
        return persona_profile

    def _generate_response_llama(self, prompt: str) -> str:
        """
        Generates a response using a Llama model.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The generated response.

        Raises:
            Exception: If response generation fails.
        """
        try:
            result = self.text_model(prompt)
            return result['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Llama response generation error: {e}")
            return "I encountered an error generating a response."

    def _generate_response(self, prompt: str) -> str:
        """
        Generates a response using a Hugging Face Transformers model.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The generated response.

        Raises:
            Exception: If response generation fails.
        """
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.text_model.generate(
                inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I encountered an error generating a response."

    def process_interaction(self, user_message: str) -> str:
        """
        Processes a user interaction, generating a response that is consistent
        with the persona defined in the MindFile.

        Args:
            user_message (str): The user's input message.

        Returns:
            str: The generated response.
        """
        cached_response = self.response_cache.get_cached_response(user_message)
        if cached_response:
            return cached_response

        #  Crucially, we now inject persona information into the prompt.
        persona_context = self._build_persona_context()
        enhanced_prompt = (
            f"Context: {persona_context}\n"
            f"User Query: {user_message}\n"
            "Respond as if you were the person described in the context, maintaining their personality, style, and beliefs:"
        )

        response = self._generate_response(enhanced_prompt)
        self.response_cache.cache_response(user_message, response)
        return response

    def _build_persona_context(self) -> str:
        """
        Constructs a context string that encapsulates the persona's key attributes
        extracted from the MindFile.

        Returns:
            str: A string representing the persona's context.
        """
        #  This method formats the persona profile into a string that can be
        #  injected into the prompt.  It needs to be adapted based on how
        #  `_extract_persona_profile` is implemented.
        return "\n".join([f"{key}: {value}" for key, value in self.persona_profile.items()])


class InfiniteAttentionEngine:
    def __init__(
        self,
        mindfile_path: str = "MindFile.txt",
        model_path: Optional[str] = "../Models/gemma-3-1b-it-Q8_0.gguf",
        model_name: str = "microsoft/DialoGPT-medium",
        embedding_model: str = 'all-MiniLM-L6-v2',
        cache_db_path: str = "infinite_attention_cache.db",
        chunk_size: int = 512,
        max_workers: int = 4  # Added max_workers parameter
    ):
        # ... [previous __init__ method remains the same]
        self.max_workers = max_workers
        self.processing_details = []  # New attribute to track processing details

    def split_mindfile(self, text: str, chunk_size: int = 512) -> List[str]:
        """
        Splits the mindfile text into overlapping chunks.

        Args:
            text (str): The entire mindfile text
            chunk_size (int): Size of each chunk

        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        for i in range(0, len(text), chunk_size // 2):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def process_chunk(self, chunk: str, user_message: str, extracted_facts: Dict[str, str]) -> str:
        """
        Processes a single chunk of the mindfile.

        Args:
            chunk (str): A text chunk from the mindfile
            user_message (str): The user's input message
            extracted_facts (Dict[str, str]): Important facts extracted from the mindfile

        Returns:
            str: A response generated based on the chunk and context
        """
        # Inject context from extracted facts and the current chunk
        context_prompt = (
            f"Context from MindFile:\n{' '.join([f'{k}: {v}' for k, v in extracted_facts.items()])}\n\n"
            f"Chunk content: {chunk}\n\n"
            f"User Query: {user_message}\n"
            "Respond relevantly:"
        )

        return self._generate_response(context_prompt)

    def filter_results(self, responses: List[str], threshold: float = 0.5) -> List[str]:
        """
        Filters responses based on relevance.

        Args:
            responses (List[str]): List of generated responses
            threshold (float): Relevance threshold

        Returns:
            List[str]: Filtered list of responses
        """
        # This is a placeholder. In a real implementation, you'd use semantic similarity
        # to determine the relevance of each response
        return [resp for resp in responses if len(resp.strip()) > 10]

    def aggregate_responses(self, responses: List[str], user_message: str) -> str:
        """
        Aggregates multiple responses into a coherent final response.

        Args:
            responses (List[str]): List of filtered responses
            user_message (str): The original user message

        Returns:
            str: An aggregated, synthesized response
        """
        # Simple aggregation by concatenating unique responses
        unique_responses = list(set(responses))
        aggregated_context = f"User Query: {user_message}\n"
        aggregated_context += "Synthesized Responses:\n" + "\n".join(unique_responses)

        return self._generate_response(
            f"{aggregated_context}\n\n"
            "Generate a concise, coherent response that captures the essence of these insights:"
        )

    def process_interaction(self, user_message: str) -> str:
        """
        Enhanced interaction processing with multi-chunk analysis.

        Args:
            user_message (str): The user's input message

        Returns:
            str: The generated response
        """
        # Reset processing details for this interaction
        self.processing_details = []

        # Check cache first
        cached_response = self.response_cache.get_cached_response(user_message)
        if cached_response:
            return cached_response

        # Extract facts from the entire MindFile
        extracted_facts = self.persona_profile

        # Split MindFile into chunks
        chunks = self.split_mindfile(self.mindfile_text, self.chunk_size)
        responses = []

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self.process_chunk, chunk, user_message, extracted_facts): chunk
                for chunk in chunks
            }

            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    response = future.result()
                    self.processing_details.append({
                        'Fragment': chunk[:50] + '...',
                        'Response': response
                    })
                    responses.append(response)
                except Exception as e:
                    self.processing_details.append({
                        'Fragment': chunk[:50] + '...',
                        'Response': f'Error: {e}'
                    })

        # Filter and aggregate responses
        valid_results = self.filter_results(responses)
        final_output = self.aggregate_responses(valid_results, user_message)

        # Cache and return the response
        self.response_cache.cache_response(user_message, final_output)
        return final_output

def main():
    """
    Modified main function to display processing details.
    """
    try:
        engine = InfiniteAttentionEngine()
    except Exception as e:
        logger.error(f"Engine initialization failed: {e}")
        sys.exit(1)

    print("🤖 Infinite Attention Engine Initialized. Type 'exit' to quit.")

    while True:
        try:
            start_time = time.time()
            process = psutil.Process(os.getpid())

            user_input = input("User: ")

            if user_input.lower() in ['exit', 'bye', 'quit']:
                print("System: Goodbye!")
                break

            response = engine.process_interaction(user_input)
            print("System:", response)

            end_time = time.time()
            print(
                f"⏳ Cognitive Cost: "
                f"Time: {end_time - start_time:.2f}s | "
                f"🔹 CPU: {psutil.cpu_percent()}% | "
                f"RAM: {process.memory_info().rss / (1024 * 1024):.2f} MB"
            )

            # Display processing details as a table
            if engine.processing_details:
                print("\nProcessing Details:")
                df = pd.DataFrame(engine.processing_details)
                print(df)
            else:
                print("\nNo processing details recorded for this turn.")

            # Existing database interaction log display
            if db_interaction_log:
                print("\nDatabase Interaction Log:")
                df = pd.DataFrame(db_interaction_log)
                print(df)
                db_interaction_log.clear()

        except KeyboardInterrupt:
            print("\nSystem: Interrupted. Type 'exit' to quit.")
        except Exception as e:
            logger.error(f"Interaction error: {e}")
            print("System: An unexpected error occurred.")

if __name__ == "__main__":
    main()

