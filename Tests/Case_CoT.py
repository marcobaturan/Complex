###############################################################################
#                          Llama Model Interaction                            #
#                                                                             #
# This script interacts with a local Llama-based LLM to generate responses.   #
# It supports both direct responses and Chain of Thought (CoT) reasoning.     #
#                                                                             #
# Features:                                                                   #
# - Basic response generation.                                                #
# - Chain of Thought (CoT) and Auto-CoT reasoning with step-by-step logic.    #
# - Supports both greedy and sampling strategies for response generation.     #
###############################################################################

# ========================== Import Required Modules ==========================
try:
    import sys
    import time as t
    import traceback
    from time import perf_counter  # For precise execution time measurement
    from llama_cpp import Llama
    from functools import lru_cache  # Importing functools for memoization
    print("All modules loaded.")
except Exception as e:
    print("Module loading error:", str(e))
    traceback.print_exc()
    sys.exit(1)

# ======================== Load the Llama Model ================================
def load_llama_model(model_path, n_threads=8, n_gpu_layers=35):
    """Load the Llama model with specified parameters."""
    try:
        # Inicializa el modelo Llama con los parámetros proporcionados
        llm = Llama(
            model_path=model_path,  # Ruta del modelo GGUF
            n_ctx=2048,  # Tamaño del contexto (longitud de la ventana de atención)
            n_threads=n_threads,  # Número de hilos de CPU para la inferencia
            n_gpu_layers=n_gpu_layers,  # Número de capas que se ejecutan en la GPU
            logits_all=False  # Devuelve solo el último token (optimiza el rendimiento)
        )
        print("Model loaded successfully.")
        return llm
    except Exception as e:
        print("Error loading model:", str(e))
        traceback.print_exc()
        sys.exit(1)

# ======================= Basic Response Generation ============================
@lru_cache(maxsize=128)  # Decorator for caching up to 128 responses
def generate_response(llm, prompt, strategy="greedy"):
    """Generate a response using the specified decoding strategy (greedy or sampling)."""
    try:
        start_time = perf_counter()  # Marca el inicio del tiempo de ejecución

        # Formatea el mensaje en el formato esperado por Llama
        messages = [
            {"role": "system", "content": "You are a general assistant."},
            {"role": "user", "content": prompt}
        ]

        # Selecciona la estrategia de decodificación
        if strategy == "greedy":
            # Estrategia "greedy": sin aleatoriedad (temperatura = 0.0)
            output = llm.create_chat_completion(
                messages=messages,
                max_tokens=512,  # Límite de longitud de respuesta
                stop=["\n", "User:", "###"],  # Tokens de parada
                temperature=0.0  # Determinismo total
            )
        elif strategy == "sampling":
            # Estrategia "sampling": con aleatoriedad controlada (temperature y top_p)
            output = llm.create_chat_completion(
                messages=messages,
                max_tokens=512,  # Límite de longitud de respuesta
                stop=["\n", "User:", "###"],  # Tokens de parada
                temperature=0.7,  # Mayor temperatura aumenta la creatividad
                top_p=0.9  # Nucleus sampling (para limitar la probabilidad acumulativa)
            )
        else:
            raise ValueError("Invalid strategy. Use 'greedy' or 'sampling'.")

        end_time = perf_counter()  # Marca el fin del tiempo de ejecución

        # Extrae y devuelve la respuesta generada
        response = output['choices'][0]['message']['content']
        print(f"Response time: {end_time - start_time:.2f} seconds")
        return response

    except Exception as e:
        print("Error generating response:", str(e))
        traceback.print_exc()
        return ""

# ================= Chain of Thought (CoT) Response Generation =================
@lru_cache(maxsize=128)  # Decorator for caching up to 128 responses
def generate_cot_response(llm, prompt, auto_cot=False, strategy="greedy"):
    """Generate a Chain of Thought (CoT) response with optional Auto-CoT."""
    try:
        # Añade la instrucción para el razonamiento paso a paso
        cot_prompt = f"{prompt}\nThink step by step."
        if auto_cot:
            # Si se activa auto_cot, guía el modelo con sugerencias adicionales
            cot_prompt += "\nUse analogies and common patterns to guide your reasoning."

        print("CoT prompt:", cot_prompt)

        # Genera la respuesta usando la función principal
        return generate_response(llm, cot_prompt, strategy)

    except Exception as e:
        print("Error generating CoT response:", str(e))
        traceback.print_exc()
        return ""

# ============================= Main Execution ================================
if __name__ == "__main__":
    try:
        # Ruta relativa al modelo desde el directorio actual
        model_path = "../Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

        # Carga el modelo
        llm = load_llama_model(model_path)

        # Prompt de ejemplo para probar el modelo
        prompt = "Solve the following math problem: If John has 7 pencils and gives away 2, how many are left?"
        print("User prompt:", prompt)

        # Genera una respuesta usando la estrategia greedy (determinística)
        response = generate_response(llm, prompt, strategy="greedy")
        print("Model Response (Greedy):", response)

        # Genera una respuesta usando Chain of Thought (CoT) con muestreo (sampling)
        cot_response = generate_cot_response(llm, prompt, auto_cot=True, strategy="sampling")
        print("CoT Response (Sampling):", cot_response)

    except Exception as e:
        print("Error during execution:", str(e))
        traceback.print_exc()
        sys.exit(1)
