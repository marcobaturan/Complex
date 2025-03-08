# Tree of Thought Implementation with Optimisations
# ------------------------------------------------
# This module implements a Tree of Thought reasoning approach using the Llama model.
# The implementation features comprehensive memoisation and optimisation strategies
# to enhance performance whilst maintaining reasoning quality.

import sys  # For system-specific parameters and functions
import time  # For performance measurement
import functools  # For advanced function decorators
from llama_cpp import Llama  # For LLM interactions via Llama

# Record the starting time to measure overall execution performance
start_time = time.time()


# Advanced memoisation decorator with optional timeout to prevent stale cache entries
def memoize(func=None, *, timeout=None):
    """
    A sophisticated memoisation decorator that caches function results.

    This implementation supports an optional timeout parameter to invalidate
    cache entries after a specified period, ensuring freshness of data.

    Args:
        func: The function to be memoized
        timeout: Optional time in seconds after which cache entries expire

    Returns:
        The memoized function with caching capabilities
    """
    if func is None:
        return lambda f: memoize(f, timeout=timeout)

    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a hashable key from both positional and keyword arguments
        key = (args, frozenset(kwargs.items()) if kwargs else None)
        current_time = time.time()

        # Check if result exists in cache and hasn't expired
        if key in cache:
            result, timestamp = cache[key]
            if timeout is None or current_time - timestamp < timeout:
                return result

        # Calculate result if not in cache or expired
        result = func(*args, **kwargs)
        cache[key] = (result, current_time)
        return result

    # Add function to clear cache when needed
    wrapper.clear_cache = cache.clear
    return wrapper


class ThoughtNode:
    """
    Represents a single node in the Tree of Thoughts.

    Each node contains a thought (text) and may have multiple children
    representing the next possible thoughts in the reasoning chain.
    """

    __slots__ = ['thought', 'children', 'score']

    def __init__(self, thought, score=0.0, children=None):
        """
        Initialise a thought node with the given parameters.

        Args:
            thought (str): The textual content of this thought
            score (float): Optional quality/relevance score for this thought
            children (list): Optional list of child ThoughtNodes
        """
        self.thought = thought  # The textual content of this thought
        self.score = score  # Quality score for prioritisation
        self.children = children or []  # Child nodes, defaulting to empty list


class TreeOfThought:
    """
    Implements the Tree of Thought reasoning approach.

    This class manages the generation, exploration and evaluation of
    thought chains to solve complex reasoning problems.
    """

    def __init__(self, root_prompt, llm, max_iterations=3, max_tokens=100,
                 temperature=0.5, beam_width=2):
        """
        Initialise the Tree of Thoughts with configuration parameters.

        Args:
            root_prompt (str): The initial prompt to begin reasoning from
            llm: The language model instance to use for generating thoughts
            max_iterations (int): Maximum depth of thought tree to explore
            max_tokens (int): Maximum tokens for each LLM response
            temperature (float): Sampling temperature for the LLM
            beam_width (int): Number of branches to maintain at each level
        """
        self.root = ThoughtNode(root_prompt)  # Root of the thought tree
        self.max_iterations = max_iterations  # Maximum exploration depth
        self.llm = llm  # Language model for thought generation
        self.current_thoughts = [self.root]  # Current frontier of thoughts
        self.max_tokens = max_tokens  # Token limit for LLM calls
        self.temperature = temperature  # Control randomness in generation
        self.beam_width = beam_width  # Number of branches to maintain
        self.call_count = 0  # Counter for LLM calls (for efficiency tracking)

    @memoize(timeout=3600)  # Cache LLM responses for 1 hour
    def call_llm(self, prompt):
        """
        Communicate with the language model to generate responses.

        This method is memoized to avoid repeated identical calls to the LLM,
        significantly improving performance for repeated or similar prompts.

        Args:
            prompt (str): The prompt to send to the language model

        Returns:
            str: The generated response from the language model
        """
        try:
            self.call_count += 1  # Increment call counter for performance analysis

            # Construct the chat message format expected by Llama
            messages = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a logical reasoning assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract the model's response
            return messages['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error in LLM call: {e}")
            return ""  # Return empty string on error to allow graceful continuation

    def explore_thoughts(self, thought_nodes):
        """
        Expand the current thought frontier by generating new thoughts.

        For each thought in the current frontier, generate multiple new thoughts
        and select the most promising ones based on relevance and diversity.

        Args:
            thought_nodes (list): List of ThoughtNode objects to expand

        Returns:
            list: New frontier of ThoughtNode objects
        """
        new_thought_nodes = []

        for thought_node in thought_nodes:
            # Construct a prompt that guides the LLM to generate relevant next thoughts
            prompt = (f"Given the current thought: '{thought_node.thought}', "
                      f"provide exactly {self.beam_width} distinct next thoughts "
                      f"that logically progress this reasoning. Each thought should "
                      f"be on a separate line and should be concise yet insightful.")

            # Call the LLM to generate next thoughts
            response = self.call_llm(prompt)

            if response:
                # Process the response to extract individual thoughts
                thoughts = [t.strip() for t in response.strip().split("\n")
                            if t.strip() and len(t.strip()) > 10]

                # Ensure we only take the specified beam width number of thoughts
                thoughts = thoughts[:self.beam_width]

                # Score and create nodes for each new thought
                for i, new_thought in enumerate(thoughts):
                    # Create a new thought node
                    # (In a more advanced implementation, we would score thoughts here)
                    new_thought_node = ThoughtNode(
                        new_thought,
                        score=1.0 - (i * 0.1)  # Simple heuristic scoring
                    )

                    # Link to parent and add to frontier
                    thought_node.children.append(new_thought_node)
                    new_thought_nodes.append(new_thought_node)

        # Sort by score and limit to beam width * number of parent nodes
        # to maintain a reasonable frontier size
        new_thought_nodes.sort(key=lambda node: node.score, reverse=True)
        max_frontier_size = self.beam_width * len(thought_nodes)
        return new_thought_nodes[:max_frontier_size]

    def run(self, verbose=True):
        """
        Execute the main thought exploration process.

        This iteratively expands the thought frontier up to the maximum
        iteration depth, building a tree of reasoning paths.

        Args:
            verbose (bool): Whether to print progress information
        """
        iteration = 0

        # Continue until we've reached max depth or have no thoughts to expand
        while self.current_thoughts and iteration < self.max_iterations:
            if verbose:
                print(f"Iteration {iteration + 1}:")

            # Explore and expand current thoughts
            self.current_thoughts = self.explore_thoughts(self.current_thoughts)

            # Display progress if verbose mode is enabled
            if verbose:
                for i, thought_node in enumerate(self.current_thoughts):
                    print(f"{i + 1}. {thought_node.thought} (score: {thought_node.score:.2f})")
                print(f"LLM calls so far: {self.call_count}")

            iteration += 1

        if verbose:
            print(f"Exploration complete after {iteration} iterations and {self.call_count} LLM calls.")

    def print_tree(self, node=None, level=0, max_depth=None):
        """
        Recursively print the tree structure for visualization.

        Args:
            node: The current node to print (defaults to root if None)
            level (int): Current depth in the tree for indentation
            max_depth (int): Optional maximum depth to print
        """
        if node is None:
            node = self.root

        if max_depth is not None and level > max_depth:
            return

        # Calculate indentation based on level
        indent = '  ' * level

        # Print the current node with appropriate indentation
        print(f"{indent}📝 {node.thought}")

        # Recursively print children
        for i, child in enumerate(node.children):
            self.print_tree(child, level + 1, max_depth)

    def extract_best_path(self):
        """
        Extract the highest-scoring complete path through the tree.

        Returns:
            list: Sequence of thoughts representing the best reasoning path
        """
        # This is a simplified implementation that could be enhanced with
        # more sophisticated path selection algorithms

        best_path = []
        current = self.root
        best_path.append(current.thought)

        while current.children:
            # Select the highest-scoring child
            current = max(current.children, key=lambda node: node.score)
            best_path.append(current.thought)

        return best_path


def main():
    """
    Main execution function to demonstrate the Tree of Thought approach.
    """
    try:
        # Initialise the language model with optimised parameters
        llm = Llama(
            model_path="../Models/TinyLlama-R1-LIMO.Q4_K_S.gguf",
            n_ctx=2048,  # Context length: 0.75 word per token, approx. 1500 words
            n_threads=8,  # Use multithreading for better performance
            n_gpu_layers=35  # Offload computation to GPU where possible
        )

        print("Language model initialised successfully.")

        # Define the problem statement
        starting_prompt = """You are an expert in formal logic and deductive reasoning. Below, you are presented with a complex problem that requires multiple steps of logical inference. You must answer clearly, justifying each step of your reasoning.

Problem: In a city, there are three types of people:
- **Truthful**: They always tell the truth.  
- **Liars**: They always lie.  
- **Random**: They respond randomly.  

You come across three people: A, B, and C. You know that each belongs to a different type (one is truthful, one is a liar, and one is random), but you don't know who is who.

You ask the following questions:
1. Statement to A: 'Are you the truthful one?' - Answer, 'Yes.'  
2. Assertion to B: 'Would A say you are the random one?' - Answer, 'No.'  
3. Statement to C: 'If I ask you if B is the liar, what would you answer?' - Answer, 'Yes.'  

Assignments:
1. Identify who is the truthful one, who is the liar, and who is the random one.
2. Explain step by step how you came to that conclusion.
3. Consider any ambiguity or cheating in the answers.

**Additional rules**: 
- Reason as if the model could be flawed logically, pointing out any possible flaws or paradoxes.
- Do not make unwarranted assumptions.
- If there is more than one possible answer, analyze and evaluate each with its logical probability.
"""

        # Create and run the Tree of Thought
        tot = TreeOfThought(
            root_prompt=starting_prompt,
            llm=llm,
            max_iterations=4,  # Increased from 3 for deeper reasoning
            max_tokens=150,  # Increased for more detailed thoughts
            temperature=0.4,  # Slightly reduced for more focused reasoning
            beam_width=3  # Increased from 2 for more exploration
        )

        # Execute the thought exploration process
        tot.run()

        # Display results
        print("\n" + "=" * 80)
        print("Complete Tree of Thoughts:")
        tot.print_tree()

        print("\n" + "=" * 80)
        print("Best Reasoning Path:")
        best_path = tot.extract_best_path()
        for i, thought in enumerate(best_path):
            print(f"{i}. {thought}")

    except Exception as e:
        print(f"Error in execution: {e}")

    finally:
        # Report performance metrics
        execution_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"Total execution time: {execution_time:.2f} seconds")

        # If the TreeOfThought object exists, report LLM call statistics
        if 'tot' in locals():
            print(f"Total LLM calls: {tot.call_count}")
            print(f"Average time per call: {execution_time / tot.call_count:.2f} seconds")


# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()