# Complex

## Abstract
    Complex is a framework for research in the field of artificial intelligence.
    Complex is a Cognitive Architecture (CA) built from compact multi-module multimodal (4M)
    multimodal models (4M). The modules behave like the bricks of the CA and are linked together by modular correlates.
    by means of modular correlates, symbolically mimicking the concept of neural correlates.
    
    The project is divided into Tests, Models and Agents. It starts with the Tests, acting as an AI lab to test the 
    AI laboratory to test the different models and improve them through programmatic cognitive strategies such as Tree 
    programmatic cognitive strategies such as Tree-of-Thoughts (ToT) or Chain-of-Thoughts (CoT). And then 
    more complex cognitive structures will be tested to evaluate their performance in context.

    Finally, they will be assembled within the Agent folder. Where an evaluation of the agent's cognitive
    cognitive activity of the agent and a record of it will be kept.

    The ultimate goal is to test increasingly complex and efficient architectures that can be run on computers of limited capacity, achieving 
    on computers of limited capacity, achieving virtual models of artificial brains.

## Resumen
    Complex es un marco de trabajo para la investigación en el campo de la inteligencia artificial.
    Complex es una arquitectura cognitiva (AC) construida a partir de módulos multiples de modelos
    multimodales (4M) compactos. Los modulos se comportan como los ladrillos de la AC y estos se unen
    por medio de correlatos modulares, imitando simbólicamente el concepto de correlato neuronal.

    El proyecto se divide en Tests, Modelos y Agentes. Donde se comienza por los Tests, haciendo de 
    laboratorio de IA para poner a prueba los distintos modelos y mejorarlos mediantes estrategías 
    cognitivas programáticas tales como Tree-of-Thoughts (ToT) o Chain-of-Thoughts (CoT). Y después 
    se probarán más estructuras cognitivas complejas para evaluar su desempeño en el contexto.

    Finalmente, se ensamblarán dentro de la carpeta Agente. Dónde se hará una evaluación de la actividad
    cognitiva del agente y se llevará un registro del mismo.
    
    El objetivo final es probar arquitecturas cada vez más complejas y eficientes que puedan ejecutarse 
    en ordenadores de capacidad limitados, logrando modelos virtuales de cerebros artificiales.

## Summaries

--------------------------------------------------------

**Summary of LLM Tree of Thoughts (LLM_ToT) Code**

### Objective:
The LLM Tree of Thoughts (LLM_ToT) code is designed to explore and expand logical reasoning through iterative thought generation using a Large Language Model (LLM). It simulates a structured thinking process by representing ideas as nodes in a tree and expanding these ideas through multiple iterations of model-driven prompts. This implementation can be used to solve complex problems step-by-step by generating and evaluating potential solutions.

### Key Components and Their Roles:

1. **Module Imports:**
   - `sys`: Provides access to system-specific parameters and functions.
   - `time` (aliased as `t`): Measures execution time to track performance.
   - `llama_cpp`: Interfaces with the Llama model for generating responses.

2. **Performance Measurement:**
   - The code records the start time (`start = t.time()`) to calculate the total execution duration later.

3. **ThoughtNode Class:**
   - Represents a node in the thought tree.
   - Attributes:
      - `thought`: Stores the content of the idea.
      - `children`: Holds the list of subsequent thought nodes (default is an empty list).

4. **TreeOfThought Class:**
   This is the core structure managing thought generation and exploration.
   
   **Attributes:**
   - `root`: Root node of the thought tree initialized with the starting prompt.
   - `max_iterations`: Limits the depth of thought exploration (default is 3).
   - `llm`: Instance of the Llama model for generating responses.
   - `current_thoughts`: Tracks active thoughts to be expanded.
   - `max_tokens`: Controls the output length of each LLM-generated response.

   **Methods:**

   - `__init__`: Initializes the tree with a starting prompt and parameters.

   - `call_llm`: Interfaces with the Llama model to generate responses based on provided prompts.
      - Uses a structured message format for the LLM.
      - Employs a temperature value of 0.5 for balanced randomness and determinism.
      - Handles errors gracefully and returns an empty string in case of failure.

   - `explore_thoughts`: Expands the tree by generating new thoughts from the LLM.
      - Constructs a prompt to guide the LLM in producing two new ideas.
      - Parses the LLM's output and appends new thoughts as child nodes.

   - `run`: Executes the iterative thought exploration process.
      - Loops through the thought tree, expanding each node up to `max_iterations`.
      - Prints each explored thought for transparency.

   - `update_starting_thought`: Allows the root thought to be updated dynamically, resetting the tree for new exploration.

   - `print_tree`: Recursively displays the hierarchical structure of the tree, representing each thought and its descendants.

5. **Llama Model Initialization:**
   - The model is instantiated with specific parameters:
      - `model_path`: Path to the Llama model file.
      - `n_ctx`: Context length (2048 tokens, approximately 1500 words).
      - `n_threads`: Number of CPU threads (8 for parallelism).
      - `n_gpu_layers`: Number of model layers offloaded to GPU (35 for acceleration).

6. **Execution Flow:**
   - The script starts by defining a complex logical reasoning problem as the initial prompt.
   - An instance of `TreeOfThought` is created with the prompt and Llama model.
   - The `run()` method initiates the thought generation and exploration process.
   - Upon completion, the entire tree structure is displayed using `print_tree()`.

7. **Performance Reporting:**
   - After execution, the script calculates and prints the total time taken to process the thought tree (`end - start`).

### Example Problem:
The script includes a sample logical deduction problem involving three individuals (Truthful, Liar, Random) and their responses to specific questions. The model iteratively explores and analyzes potential solutions to identify each person's identity while considering logical ambiguities and inconsistencies.

### Use Case:
This implementation is suitable for:
- Solving complex logical and deductive reasoning problems.
- Exploring decision trees and generating multi-step solutions.
- Analyzing ambiguous or multi-outcome scenarios.
- Enhancing cognitive models through iterative idea expansion.
---------------------------------------------------------------------------------


## Models/Modelos

- [TinyLlama-R1-LIMO-GGUF](https://huggingface.co/mradermacher/TinyLlama-R1-LIMO-GGUF/tree/main)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)
- [Moondream](https://moondream.ai/)
- [storyboard-sketch](https://huggingface.co/blink7630/storyboard-sketch)

## Sources/Fuentes

- [SQL & LLM](https://medium.com/version-1/enhancing-database-querying-using-large-language-models-69a2064bae6b)
- [PoT](https://learnprompting.org/docs/advanced/decomposition/program_of_thoughts?srsltid=AfmBOorJbY006Y3fjUwy2cAI3kAte1_TOzNdDAod04sGe4SCOnCnjtJx)
- [CoT](https://medium.com/@sujathamudadla1213/chain-of-thought-cot-tree-of-thought-tot-and-react-response-act-6d8103f52a48)
- [SoT](https://www.analyticsvidhya.com/blog/2024/07/skeleton-of-thoughts/)
- [ToT](https://www.ibm.com/es-es/topics/tree-of-thoughts#:~:text=Marco%20para%20el%20%C3%A1rbol%20de,a%20los%20procesos%20cognitivos%20humanos)

## References/Referencias

- [TinyLlama installation](https://dev.to/_ken0x/tinyllama-llm-a-step-by-step-guide-to-implementing-the-11b-model-on-google-colab-1pjh)
- [ToT implementation](https://dev.to/stephenc222/how-to-implement-a-tree-of-thoughts-in-python-4jmc)

## Tools and sources of consultation/Herramientas y fuentes de consulta

- AIMA (Artificial Intelligence: A Modern Approach) 3º Edition for symbolic cognition.
- GPT and Claude for PDF reports.
