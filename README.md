# Complex

## Authors

- [Marco Baturan](https://github.com/marcobaturan)
- [Roman Sitelew](https://github.com/RomanPlusPlus)

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
# TinyLlama Model Load and Execution Report

## Summary

The `tinyllama-1.1b-chat-v1.0` model has been successfully loaded in the environment. This model has been optimized to enhance its performance by leveraging the **Chain of Thought (CoT)** technique, which allows for better utilization of the model’s capacity by breaking down complex problems into simpler steps and logical reasoning. The model's load and execution show efficient response times, with improved evaluation times through the use of CoT.

## Load Details

- **Model Loaded**: `tinyllama-1.1b-chat-v1.0`
- **Model Size**: 636.18 MiB
- **Quantization Version**: Q4_K (Optimized for medium size)
- **Architecture**: Llama
- **Special Tokens**:
  - BOS Token: `<s>`
  - EOS Token: `</s>`
  - UNK Token: `<unk>`
  - PAD Token: `</s>`

## Optimization Description

**Chain of Thought (CoT)** has been implemented to maximize the model's reasoning ability, allowing it to break down complex tasks into step-by-step reasoning. This enhances the model's ability to provide more accurate and detailed responses by performing thorough evaluations rather than simple or direct answers.

### CoT Example

**Prompt**: "Solve the following math problem: If John has 7 pencils and gives away 2, how many are left?"

**CoT Response**:  
- Step 1: John has 7 pencils.
- Step 2: He gives 2 pencils to someone else.
- Step 3: Subtract 2 from 7.
- Result: John has 5 pencils.

The model uses reasoning based on analogies and common patterns to approach the problem, improving the clarity and accuracy of the responses.

## Performance

- **Load Time**: 447.14 ms
- **Evaluation Time (Prompt)**: 447.04 ms / 57 tokens (7.84 ms per token)
- **Evaluation Time (Response)**: 940.64 ms / 45 executions (20.90 ms per token)
- **Total Time**: 1403.50 ms / 102 tokens

**Response Time**: 1.40 seconds

## Conclusion

The implementation of CoT, along with optimizations in model load and execution, has demonstrated an improvement in performance and the model's ability to process and generate more complex and detailed responses. Using this technique has allowed for maximizing the model's efficiency and improving its accuracy in reasoning tasks.

# Conclusive Report on the Skeleton of Thoughts (SoT) Use Case

## Summary of the Use Case

The Skeleton of Thoughts (SoT) use case involved the implementation of a local language model, specifically `tinyllama-1.1b-chat-v1.0`, to generate and expand a hierarchical outline on the topic "The Impact of Artificial Intelligence on Future Job Markets." The goal was to demonstrate the model's ability to create coherent and detailed thought structures, as well as to expand specific points within the outline.

## Model Setup and Loading

- **Model Used**: `tinyllama-1.1b-chat-v1.0`
- **Architecture**: Llama
- **Maximum Context**: 2048 tokens
- **Model Layers**: 22
- **Attention Heads**: 32
- **Embedding Length**: 2048
- **Model Size**: 1.10B parameters

The model was successfully loaded with 23 key-value pairs and 201 tensors, using the GGUF V3 version. The model loading and tokenizer initialization were completed without issues, with efficient resource allocation to the CPU.

## Initial Outline Generation

The initial outline was generated correctly with a depth of 2 levels and 3 points per level. The model provided a coherent structure that addressed various aspects of AI's impact on the job market, including the creation of new opportunities and the need for training programs.

## Expansion of Specific Points

Points "1." and "2.1." were expanded to provide a more detailed analysis. The expansion of point "1." offered deeper insights into how AI can affect different industries, while the expansion of point "2.1." focused on the importance of AI adoption and upskilling for employees.

## Addition of Cross-Links

Cross-links were added between points "1." and "2.", and between "1.1." and "2.1." to establish meaningful relationships within the outline. These links helped connect ideas and provide a more integrated view of AI's impact on the job market.

## Summary Generation

The attempt to generate a summary of the outline was unsuccessful due to the lack of content in the outline at that moment. This suggests that the model needs a more complete outline to generate meaningful summaries.

## Outline Export

The final outline was successfully exported to a Markdown file (`skeleton_output.md`), facilitating its viewing and further editing.

## Performance

- **Total Execution Time**: 37.95 seconds
- **API Calls**: 3
- **Total Tokens**: 1920
- **Outline Size**: 13 points
- **Cache Hits**: 3
- **Average Time per Call**: 12.65 seconds
- **Average Tokens per Call**: 640

The overall performance was efficient, with appropriate use of caching to optimize repeated calls.

## Conclusion

The use case demonstrated the `tinyllama-1.1b-chat-v1.0` model's capability to generate and expand thought structures coherently and in detail. However, areas for improvement were identified, such as the need for a more complete outline for summary generation and the optimization of the model interface to avoid argument errors. Overall, the model showed promising performance for applications requiring the generation of complex thought structures.

### Rethink Algorithm: Failure Analysis Report

**Objective:**
Evaluate the performance of the Rethink algorithm when processing complex prompt refinement tasks.

**Input Prompt:**
"Design a sustainable city for the year 2050 that optimizes for environmental conservation, efficient transportation, and equitable access to resources. Consider advanced technologies like AI governance, renewable energy systems, and circular economies. Describe how the city balances privacy, security, and personal freedom while maintaining a high quality of life."

**Output (Erroneous Result):**
A list of unrelated "Sunday Night" music segments focusing on various genres such as country, world pop, electronic, instrumental, and variety.

**Identified Issues:**
1. **Context Misalignment:**
   - The algorithm produced output unrelated to the original prompt.
   - Failure to maintain thematic relevance in the refinement process.

2. **Semantic Drift:**
   - The output shifted focus from urban design to a music-based content structure.

3. **Execution Anomaly:**
   - Despite the complexity of the input, the algorithm produced a simplistic, repetitive output unrelated to the task.

**Root Cause Analysis:**
- Misclassification of input context leading to an incorrect processing path.
- Overfitting to patterns of previous outputs rather than dynamically adjusting to new semantic spaces.
- Possible cache contamination or erroneous prompt routing within the algorithm's decision tree.

**Recommendations:**
1. **Context Validation Layer:**
   - Implement a validation checkpoint to verify semantic alignment during intermediate stages.

2. **Adaptive Prompt Parsing:**
   - Enhance the algorithm to dynamically adapt to complex multi-factor prompts.

3. **Error Logging & Diagnostics:**
   - Introduce comprehensive logs to track decision paths and identify similar anomalies.

**Conclusion:**
The Rethink algorithm failed to produce a relevant refined prompt due to context misalignment and semantic drift. Implementing context validation and adaptive parsing mechanisms will enhance accuracy in future iterations.

### **Experimental Report: "Writer & Critic" Prompt Refinement**  

The "Writer & Critic" in ReThink 2.0 version framework successfully refined the prompt within a single iteration. The model generated a coherent and valid response that expanded the original prompt by specifying key aspects to address (architecture, layout, sustainability, and innovation). The critic did not identify hallucinations or errors, confirming the output's validity.  

Performance metrics indicate that the model processed 660 tokens in approximately **15.34 seconds**, achieving an average rate of **42.80 tokens/second** during the evaluation phase. The use of **lru_cache** optimized repeated calls, improving efficiency with a prefix-match hit of 7 tokens.  

The experiment demonstrates that the iterative critique mechanism effectively guides the model towards clearer and more comprehensive outputs while maintaining performance. Further testing could explore multi-iteration scenarios and prompt complexity to evaluate the system's robustness under diverse conditions.

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
- [CoT1](https://clickup.com/blog/chain-of-thought-prompting/)
- [CoT2](https://medium.com/@devmallyakarar/chain-of-thought-cot-in-large-language-models-prompting-and-concise-cot-with-code-82821f9a832d)
- [Memoization](https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python)
- [Anlytics Vydia](https://www.analyticsvidhya.com/blog/2024/07/skeleton-of-thoughts/)
## References/Referencias

- [TinyLlama installation](https://dev.to/_ken0x/tinyllama-llm-a-step-by-step-guide-to-implementing-the-11b-model-on-google-colab-1pjh)
- [ToT implementation](https://dev.to/stephenc222/how-to-implement-a-tree-of-thoughts-in-python-4jmc)

## Tools and sources of consultation/Herramientas y fuentes de consulta

- AIMA (Artificial Intelligence: A Modern Approach) 3º Edition for symbolic cognition.
- GPT and Claude for PDF reports.
