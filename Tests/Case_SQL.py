try:
    # Import essential modules
    import sys
    import time as t
    import traceback
    from time import perf_counter  # High-precision time measurement
    from llama_cpp import Llama  # LLM interface
    from functools import lru_cache  # Cache optimization
    import sqlite3  # SQLite database

    print("All modules loaded.")
except Exception as e:
    print("Module loading error:", str(e))
    traceback.print_exc()
    sys.exit(1)

# ╔════════════════════════════════════════════════════════════════════╗
# ║                        Load TinyLlama Model                        ║
# ╚════════════════════════════════════════════════════════════════════╝
model_path = "../Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # Set path to your local model
llm = Llama(model_path=model_path)

# ╔════════════════════════════════════════════════════════════════════╗
# ║                 Initialize or Create SQLite Database               ║
# ╚════════════════════════════════════════════════════════════════════╝
@lru_cache(maxsize=1)  # Cache database connection for efficiency
def inicializar_db(db_name="knowledge.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create table if it does not exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS conocimientos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pregunta TEXT UNIQUE,
                        respuesta TEXT)''')

    # Populate table with initial data if empty
    cursor.execute("SELECT COUNT(*) FROM conocimientos")
    if cursor.fetchone()[0] == 0:
        datos_iniciales = [
            ("¿Cuál es la capital de Francia?", "París"),
            ("¿Quién escribió '1984'?", "George Orwell"),
            ("¿Cuál es la velocidad de la luz?", "299,792 km/s"),
        ]
        cursor.executemany("INSERT INTO conocimientos (pregunta, respuesta) VALUES (?, ?)", datos_iniciales)
        conn.commit()
        print("Database initialized with example data.")

    return conn, cursor

# ╔════════════════════════════════════════════════════════════════════╗
# ║                 Search Database for Partial Matches                ║
# ╚════════════════════════════════════════════════════════════════════╝
@lru_cache(maxsize=128)  # Cache query results for faster retrieval
def buscar_en_db(pregunta, cursor):
    cursor.execute("SELECT respuesta FROM conocimientos WHERE pregunta LIKE ?", (f"%{pregunta}%",))
    resultados = cursor.fetchall()
    return [r[0] for r in resultados] if resultados else []

# ╔════════════════════════════════════════════════════════════════════╗
# ║               Generate Response Using TinyLlama Model              ║
# ╚════════════════════════════════════════════════════════════════════╝
def generar_respuesta(prompt):
    inicio = perf_counter()  # Start timing
    output = llm(prompt, max_tokens=100)  # Generate model output
    fin = perf_counter()  # End timing

    print(f"Model response time: {fin - inicio:.2f} s")
    return output['choices'][0]['text'].strip()  # Return cleaned output

# ╔════════════════════════════════════════════════════════════════════╗
# ║       Combine Database and Model Responses for Better Accuracy     ║
# ╚════════════════════════════════════════════════════════════════════╝
def responder(pregunta):
    conn, cursor = inicializar_db()

    # Search database for existing knowledge
    respuestas_db = buscar_en_db(pregunta, cursor)

    # Generate model response
    respuesta_llm = generar_respuesta(pregunta)

    # If database has relevant information, enrich model context
    if respuestas_db:
        contexto_adicional = "\n".join([f"Verified fact: {r}" for r in respuestas_db])
        respuesta_llm = generar_respuesta(f"{contexto_adicional}\n{pregunta}")

    # Return combined output
    return f"(DB) {', '.join(respuestas_db)}\n(LLM) {respuesta_llm}"

# ╔════════════════════════════════════════════════════════════════════╗
# ║                       Interactive Query Loop                       ║
# ╚════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    while True:
        pregunta = input("Ask a question: ").strip()

        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("Exiting...")
            break

        print(responder(pregunta))
