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
    print("All modules loaded.")
except Exception as e:
    print("Some modules are missing: " + str(e))

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
            return

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
        try:
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT response FROM response_cache WHERE query_hash = ?",
                    (query_hash,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
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


def main():
    """
    Main function to run the Infinite Attention Engine in an interactive loop.
    """
    try:
        engine = InfiniteAttentionEngine()
    except Exception as e:
        logger.error(f"Engine initialization failed: {e}")
        sys.exit(1)

    print("🤖 Infinite Attention Engine Initialized. Type 'exit' to quit.")