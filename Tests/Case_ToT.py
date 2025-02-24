# Import necessary modules and libraries
try:
    import sys  # System-specific parameters and functions
    import time as t  # Time-related functions, aliasing to 't' for brevity
    from llama_cpp import Llama  # Importing Llama for LLM interactions
    print("All modules loaded.")  # Confirmation message if all modules load successfully
except Exception as e:
    print("Some modules are missing: " + str(e))  # Error message if any module fails to load

# Record the starting time for performance measurement
start = t.time()

# ThoughtNode class to represent nodes in the Tree of Thoughts
class ThoughtNode:
    def __init__(self, thought, children=None):
        # Store the current thought
        self.thought = thought
        # Initialise children nodes, defaulting to an empty list if not provided
        self.children = children or []

# TreeOfThought class to manage thought generation and exploration
class TreeOfThought:
    def __init__(self, root_prompt, llm, max_iterations=3, max_tokens=100):
        # Initialise the root of the thought tree with the provided prompt
        self.root = ThoughtNode(root_prompt)
        # Set the maximum number of exploration iterations
        self.max_iterations = max_iterations
        # Store the Llama model instance for LLM interaction
        self.llm = llm
        # Keep track of the current thoughts to be expanded
        self.current_thoughts = [self.root]
        # Limit the maximum tokens for each LLM call
        self.max_tokens = max_tokens

    def call_llm(self, prompt):
        # Communicate with the Llama model to generate responses
        try:
            messages = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a general assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,  # Lower temperature for more deterministic output
                max_tokens=self.max_tokens  # Limit output to specified token length
            )
            # Extract and return the LLM's response content
            return messages['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling LLM: {e}")  # Handle any errors during the LLM call
            return ""

    def explore_thoughts(self, thought_nodes):
        # Expand current thoughts by generating new ideas
        new_thought_nodes = []
        for thought_node in thought_nodes:
            # Construct a prompt to guide the LLM to generate new thoughts
            prompt = f"Given the current thought: '{thought_node.thought}', provide exactly two concise next thoughts that evolve this idea further."
            # Call the LLM to generate next thoughts
            response = self.call_llm(prompt)

            if response:
                # Split the LLM's response by lines to extract individual thoughts
                thoughts = response.strip().split("\n")
                if len(thoughts) >= 2:
                    # Ensure only the first two thoughts are processed
                    thoughts = thoughts[:2]
                    for new_thought in thoughts:
                        # Create a new ThoughtNode for each new thought
                        new_thought_node = ThoughtNode(new_thought.strip())
                        # Link the new thought to its parent
                        thought_node.children.append(new_thought_node)
                        # Add the new thought to the list for further exploration
                        new_thought_nodes.append(new_thought_node)
        return new_thought_nodes

    def run(self):
        # Main loop to iterate through the thought exploration process
        iteration = 0
        while self.current_thoughts and iteration < self.max_iterations:
            print(f"Iteration {iteration + 1}:")  # Display the current iteration
            # Explore and expand thoughts
            self.current_thoughts = self.explore_thoughts(self.current_thoughts)
            for thought_node in self.current_thoughts:
                print(f"Explored Thought: {thought_node.thought}")  # Output each new thought
            iteration += 1  # Increment the iteration counter

    def update_starting_thought(self, new_thought):
        # Update the root of the tree with a new starting thought
        self.root = ThoughtNode(new_thought)
        # Reset the current thoughts to the updated root
        self.current_thoughts = [self.root]

    def print_tree(self, node, level=0):
        # Recursive function to print the Tree of Thoughts
        indent = ' ' * (level * 2)  # Indentation for visualising tree levels
        print(f"{indent}- {node.thought}")  # Print the current thought
        for child in node.children:
            self.print_tree(child, level + 1)  # Recurse on child nodes

# Instantiate the Llama model
llm = Llama(model_path="../Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            n_ctx=2048,  # Context length: 0.75 word per token, approx. 1500 words
            n_threads=8,  # Number of threads to use
            n_gpu_layers=35)  # Number of layers offloaded to the GPU

# Execute the Tree of Thoughts
if __name__ == "__main__":
    # Define the initial problem statement
    starting_prompt = "Think of a solution to reduce the operational costs of your business."
    # Create an instance of the TreeOfThought with the starting prompt
    tot = TreeOfThought(starting_prompt, llm)
    # Run the thought exploration process
    tot.run()

    print("=" * 100)  # Visual separator for clarity
    print("Final Tree of Thoughts:")
    # Display the resulting Tree of Thoughts
    tot.print_tree(tot.root)

# Record the end time and calculate execution duration
end = t.time()
print('Cost of time: ', end - start)  # Output the total execution time
