import time
import functools
from llama_cpp import Llama

# ┌───────────────────────────────────────────────────────────────────────────────┐
# │ ReThink Class:                                                                │
# │ An iterative prompt refinement system using a local LLM (Llama model).        │
# │ Implements a "Writer & Critic" loop to generate and evaluate responses.       │
# │ Optimized using lru_cache for faster execution.                               │
# └───────────────────────────────────────────────────────────────────────────────┘

class ReThink:
    def __init__(self, model_path, n_ctx=2048, n_threads=8, n_gpu_layers=35, temperature=0.7):
        """
        Initialize the ReThink object with model parameters.

        Parameters:
        - model_path (str): Path to the Llama model.
        - n_ctx (int): Maximum context length (default: 2048).
        - n_threads (int): Number of CPU threads (default: 8).
        - n_gpu_layers (int): Number of layers offloaded to GPU (default: 35).
        - temperature (float): Sampling temperature for randomness (default: 0.7).
        """
        self.client = Llama(
            model_path=model_path,  # Path to the Llama model file.
            n_ctx=n_ctx,            # Maximum context length for processing.
            n_threads=n_threads,    # Number of CPU threads for parallel execution.
            n_gpu_layers=n_gpu_layers  # Number of layers to offload to GPU for acceleration.
        )
        self.temperature = temperature  # Temperature parameter controls output randomness.

    @functools.lru_cache(maxsize=128)  # Use lru_cache to memoize and speed up repeated calls.
    def execute_prompt(self, prompt, max_tokens=500):
        """
        Send a prompt to the Llama model and return the generated output.

        Parameters:
        - prompt (str): Input prompt to the model.
        - max_tokens (int): Maximum tokens to generate (default: 500).

        Returns:
        - str: Model's generated response.
        """
        try:
            response = self.client(
                prompt=prompt,               # Input prompt.
                max_tokens=max_tokens,       # Maximum output length.
                temperature=self.temperature # Randomness in sampling.
            )
            return response['choices'][0]['text'].strip()  # Extract and clean the response.
        except Exception as e:
            print(f"❌ Error executing prompt: {e}")
            return prompt  # Return the original prompt on error.

    def run(self, initial_prompt, max_iterations=5):
        """
        Execute the iterative "Writer & Critic" refinement loop.

        Parameters:
        - initial_prompt (str): The initial prompt to refine.
        - max_iterations (int): Maximum number of refinement cycles (default: 5).

        Returns:
        - str: Final refined prompt.
        """
        current_prompt = initial_prompt  # Store the current working prompt.

        for iteration in range(max_iterations):
            print(f"🔄 Iteration {iteration + 1}: {current_prompt}\n")

            # Writer role: Generate a creative and imaginative response.
            writer_prompt = f"""
            You are the Writer. Generate a creative and imaginative response to the following prompt:
            "{current_prompt}"
            """
            writer_response = self.execute_prompt(writer_prompt)

            # Critic role: Evaluate the Writer's response for hallucinations or errors.
            critic_prompt = f"""
            You are the Critic. Evaluate the following response for accuracy and hallucinations:

            Prompt: "{current_prompt}"
            Writer's Response: "{writer_response}"

            Identify any made-up information, factual errors, or nonsensical claims. If the response is valid, say: "Valid".
            """
            critic_response = self.execute_prompt(critic_prompt)

            # Output responses from both roles.
            print(f"✍️ Writer: {writer_response}\n")
            print(f"🧐 Critic: {critic_response}\n")

            # If the Critic approves the response, exit the loop.
            if "Valid" in critic_response:
                print("✅ Validated response. Stopping loop.")
                return writer_response

            # Update the prompt based on the Critic's feedback.
            current_prompt = f"""
            Refine this answer considering the Critic's feedback:
            "{writer_response}"

            Critic's Feedback:
            "{critic_response}"
            """

        return current_prompt  # Return the final refined prompt after iterations.

if __name__ == "__main__":
    start_time = time.time()  # Record the start time for performance measurement.

    # Initialize the ReThink object with the Llama model.
    rethink = ReThink(model_path="../Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

    # Input prompt to refine.
    initial_prompt = "Design a utopian city for the year 2100 that focuses on ecological balance, advanced AI governance, and human well-being."
    refined_prompt = rethink.run(initial_prompt)

    # Display the final refined prompt.
    print("\n✨ Final refined prompt:")
    print(refined_prompt)

    # Measure and print execution time.
    end_time = time.time()
    print(f"\n⏱️ Execution time: {end_time - start_time:.2f} seconds")
