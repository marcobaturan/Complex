# **Tree of Thoughts Implementation Using Llama.cpp**

## **Introduction**

This report documents the successful implementation of the "Tree of Thoughts" (ToT) methodology using the `llama_cpp` library. The objective was to generate and expand ideas systematically, simulating a non-linear exploration process through a hierarchical structure of thoughts.

The primary goal was to adapt the original Tree of Thoughts method, designed for the Anthropic Claude model, to function with a local Llama model. This implementation successfully replicates the thought generation and expansion process while providing insights into performance and execution time.

---

## **Project Overview**

The Tree of Thoughts (ToT) approach involves constructing a hierarchical structure where each node represents a distinct idea or thought. The structure evolves by iteratively prompting a language model (LLM) to generate subsequent thoughts, allowing for deep and systematic exploration of a given prompt.

This implementation utilises the `llama_cpp` library to interact with a locally hosted Llama model. The system prompts the LLM to produce two concise, progressive thoughts for each node, expanding the tree up to a specified number of iterations.

### **Technical Specifications:**

- **Language Model:** tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
- **Context Length:** 2048 tokens
- **Threads:** 8
- **GPU Layers:** 35
- **Temperature:** 0.5 (to ensure balanced creativity and determinism)
- **Maximum Iterations:** 3
- **Maximum Tokens per Thought:** 100

---

## **Implementation Breakdown**

### **1. System Setup**

The script begins by importing essential modules and verifying their availability. This ensures the environment is correctly configured before model invocation.

```python
import sys
import time as t
from llama_cpp import Llama
```

### **2. ThoughtNode Class**

This class represents each node in the thought tree. Each node contains a thought and a list of child nodes, allowing hierarchical organisation.

```python
class ThoughtNode:
    def __init__(self, thought, children=None):
        self.thought = thought
        self.children = children or []
```

### **3. TreeOfThought Class**

The core logic is encapsulated within the `TreeOfThought` class. It manages thought generation, expansion, and tree traversal.

#### Key Methods:

- **`call_llm()`**: Interacts with the Llama model to generate responses based on prompts.
- **`explore_thoughts()`**: Generates new thoughts by prompting the LLM to suggest two progressive ideas.
- **`run()`**: Conducts the iterative exploration process.
- **`update_starting_thought()`**: Updates the root node with a new starting thought.
- **`print_tree()`**: Prints the thought tree in a structured format.

### **4. Llama Model Integration**

The script initializes the Llama model with defined parameters to control performance and accuracy.

```python
llm = Llama(model_path="../Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=35)
```

### **5. Execution Process**

The script sets a starting prompt, initializes the `TreeOfThought` object, and executes the thought exploration. Upon completion, it prints the final thought tree and the execution time.

```python
if __name__ == "__main__":
    starting_prompt = "Think of a solution to reduce the operational costs of your business."
    tot = TreeOfThought(starting_prompt, llm)
    tot.run()
    print("Final Tree of Thoughts:")
    tot.print_tree(tot.root)
```

---

## **Results and Performance**

The implementation successfully generated a hierarchical tree of thoughts, with each iteration expanding the previous ideas.

### **Sample Output:**

```
Iteration 1:
Explored Thought: Analyse energy consumption and implement efficiency measures.
Explored Thought: Explore automation options for repetitive tasks.

Iteration 2:
Explored Thought: Adopt energy-efficient technologies to reduce costs.
Explored Thought: Use AI-driven processes to streamline operations.

Final Tree of Thoughts:
- Think of a solution to reduce the operational costs of your business.
  - Analyse energy consumption and implement efficiency measures.
    - Adopt energy-efficient technologies to reduce costs.
  - Explore automation options for repetitive tasks.
    - Use AI-driven processes to streamline operations.
```

### **Execution Time:**

- **Total Execution Time:** 17 seconds

This execution time reflects the efficiency of the `llama_cpp` library in handling multi-threaded token generation and GPU acceleration.

---

## **Conclusion**

The "Tree of Thoughts" implementation using `llama_cpp` successfully achieved its objective of systematically generating and expanding ideas. The system performed efficiently, producing meaningful and diverse thought pathways.

This method provides a robust framework for brainstorming, problem-solving, and creative exploration. By leveraging a local Llama model, the approach offers a cost-effective and scalable solution for iterative thought generation.

