from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import numpy as np

# Init model TinyLlama
llm = Llama(model_path="../Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", n_ctx=2048)
# Start embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Dict store embeddings of chat
conversation_history = {}

def get_contextual_response(prompt, conversation_id="default", max_history_length=5):
    """
    Genera una respuesta del modelo utilizando recuperación contextual.

ARGS: PROMPT (STR): The User Prompt. Conversation_id (STR): Unique identifier for conversation. Max_history_length
(INT): Maximum number of history inputs to consider. Returns: STR: the response generated by the model.
    """

    history = conversation_history.get(conversation_id,)
    prompt_embedding = embedding_model.encode(prompt)

    # Calculate similarities and select the relevant history
    relevant_history = []
    if history:
        history_embeddings = [h["embedding"] for h in history]
        similarities = np.dot(np.array(history_embeddings), prompt_embedding)
        most_relevant_indices = np.argsort(similarities)[::-1][:max_history_length]
        relevant_history = [history[i]["text"] for i in most_relevant_indices]

    context = "\n".join(relevant_history)
    full_prompt = context + "\nUsuario: " + prompt + "\nChatbot:"

    # Generate answer
    output = llm(full_prompt, max_tokens=256, echo=False)
    response = output["choices"][0]["text"].strip()

    # actuallize history chat
    conversation_history.setdefault(conversation_id, []).append(
        {"text": "Usuario: " + prompt + "\nChatbot: " + response, "embedding": prompt_embedding}
    )

    return response

# main loop
print("Chatbot iniciado. Escribe 'salir' para terminar.")
while True:
    user_input = input("Usuario: ")
    if user_input.lower() == "salir":
        break

    response = get_contextual_response(user_input)
    print("Chatbot:", response)