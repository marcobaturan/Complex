import os
import sys
import sqlite3
import hashlib
import threading
import logging
import time
from typing import Optional, Dict, List, Union

import numpy as np
import pandas as pd
import psutil
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    @staticmethod
    def load_model(
            model_path: Optional[str] = None,
            model_name: Optional[str] = "microsoft/DialoGPT-medium",
            use_local: bool = False
    ):
        """
        Flexible model loader supporting both Hugging Face and local GGUF models.

        Args:
            model_path (str, optional): Path to local model file
            model_name (str, optional): Hugging Face model repository name
            use_local (bool): Flag to indicate local model usage

        Returns:
            Tuple of (tokenizer, model)
        """
        if use_local and model_path and model_path.endswith('.gguf'):
            try:
                # Use llama-cpp for GGUF models
                llama_model = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_batch=256,
                    max_tokens=400,
                    verbose=False
                )
                return None, llama_model  # Return None for tokenizer when using GGUF
            except Exception as e:
                logger.error(f"GGUF model loading failed: {e}")
                raise

        # Hugging Face model loading
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            return tokenizer, model
        except Exception as e:
            logger.error(f"Hugging Face model loading failed: {e}")
            raise


class SemanticContextEngine:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic context extraction engine.

        Args:
            embedding_model (str): Sentence transformer model for embeddings
        """
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            logger.error(f"Embedding model loading failed: {e}")
            raise

    def extract_important_terms(self, text: str, num_terms: int = 10) -> List[str]:
        """
        Extract most important terms using TF-IDF.

        Args:
            text (str): Input text
            num_terms (int): Number of top terms to extract

        Returns:
            List of important terms
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
        Group semantically related terms using embeddings.

        Args:
            text (str): Input text
            important_terms (List[str]): Terms to find context for

        Returns:
            Dictionary of terms and their contextual sentences
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
    def __init__(self, cache_db_path: str = "infinite_attention_cache.db"):
        """
        Initialize SQLite-based response cache.

        Args:
            cache_db_path (str): Path to SQLite cache database
        """
        self.cache_db_path = cache_db_path
        self._init_cache_database()

    def _init_cache_database(self):
        """Initialize SQLite cache database."""
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
        Retrieve cached response for a query.

        Args:
            query (str): User query

        Returns:
            Cached response or None
        """
        try:
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT response FROM response_cache WHERE query_hash = ?", (query_hash,))
                result = cursor.fetchone()
                return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Cache retrieval error: {e}")
            return None

    def cache_response(self, query: str, response: str):
        """
        Cache a query-response pair.

        Args:
            query (str): User query
            response (str): Generated response
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
        """Clear entire response cache."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM response_cache")
                conn.commit()
            logger.info("Response cache cleared successfully.")
        except sqlite3.Error as e:
            logger.error(f"Cache clearing error: {e}")


class InfiniteAttentionEngine:
    def __init__(
            self,
            mindfile_path: str = "MindFile.txt",
            model_path: Optional[str] = "../Models/gemma-3-1b-it-Q8_0.gguf",
            model_name: str = "microsoft/DialoGPT-medium",
            embedding_model: str = 'all-MiniLM-L6-v2',
            cache_db_path: str = "infinite_attention_cache.db",
            max_workers: int = 8,
            chunk_size: int = 512
    ):
        """
        Initialize the Infinite Attention Engine.

        Args:
            mindfile_path (str): Path to the mindfile
            model_path (str, optional): Path to local model
            model_name (str): Hugging Face model name
            embedding_model (str): Sentence transformer model
            cache_db_path (str): Path to cache database
            max_workers (int): Maximum parallel processing workers
            chunk_size (int): Text chunk size
        """
        # Initialize components
        self.semantic_engine = SemanticContextEngine(embedding_model)
        self.response_cache = ResponseCache(cache_db_path)

        # Load mindfile
        self.mindfile_path = mindfile_path
        self.mindfile_text = self._load_mindfile()

        # Extract initial context
        self.extracted_facts = self._extract_facts_from_mindfile()

        # Determine if using local GGUF model
        use_local = model_path and os.path.exists(model_path) and model_path.endswith('.gguf')

        # Load text model
        try:
            self.tokenizer, self.text_model = ModelLoader.load_model(
                model_path=model_path,
                model_name=model_name,
                use_local=use_local
            )
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

        # Modify response generation for local GGUF models
        if isinstance(self.text_model, Llama):
            self._generate_response = self._generate_response_llama

        # Configuration
        self.max_workers = max_workers
        self.chunk_size = chunk_size

    def _load_mindfile(self) -> str:
        """
        Load text from mindfile.

        Returns:
            str: Mindfile contents
        """
        try:
            with open(self.mindfile_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("MindFile not found. Using empty string.")
            return ""

    def _extract_facts_from_mindfile(self) -> Dict[str, str]:
        """
        Extract important terms and context from mindfile.

        Returns:
            Dict of important terms and their context
        """
        try:
            important_terms = self.semantic_engine.extract_important_terms(self.mindfile_text)
            return self.semantic_engine.find_related_terms(self.mindfile_text, important_terms)
        except Exception as e:
            logger.error(f"Fact extraction error: {e}")
            return {}

    def _generate_response_llama(self, prompt: str) -> str:
        """
        Generate response using Llama.cpp model for GGUF files.

        Args:
            prompt (str): Input prompt

        Returns:
            str: Generated response
        """
        try:
            result = self.text_model(prompt)
            return result['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Llama response generation error: {e}")
            return "I encountered an error generating a response."

    def _generate_response(self, prompt: str) -> str:
        """
        Default response generation for Hugging Face models.

        Args:
            prompt (str): Input prompt

        Returns:
            str: Generated response
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
        Process user interaction with caching and context enrichment.

        Args:
            user_message (str): User's input message

        Returns:
            str: Generated response
        """
        # Check cache first
        cached_response = self.response_cache.get_cached_response(user_message)
        if cached_response:
            return cached_response

        # Prepare context
        context = " ".join(self.extracted_facts.values())
        enhanced_prompt = (
            f"Context: {context}\n"
            f"User Query: {user_message}\n"
            "Provide a nuanced, contextually aware response:"
        )

        # Generate response
        response = self._generate_response(enhanced_prompt)

        # Cache and return
        self.response_cache.cache_response(user_message, response)
        return response


def main():
    # Initialize engine
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

        except KeyboardInterrupt:
            print("\nSystem: Interrupted. Type 'exit' to quit.")
        except Exception as e:
            logger.error(f"Interaction error: {e}")
            print("System: An unexpected error occurred.")


if __name__ == "__main__":
    main()