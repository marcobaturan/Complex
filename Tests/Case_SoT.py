import re
import time
import functools
from llama_cpp import Llama  # Importa la clase Llama

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
    # Add function to get cache size
    wrapper.cache_size = lambda: len(cache)
    return wrapper

class SkeletonNode:
    """
    Represents a single node in the Skeleton of Thoughts.

    Each node contains a point (text) and may have multiple children
    representing subpoints in the hierarchical outline.
    """

    __slots__ = ['key', 'content', 'parent', 'children', 'metadata']

    def __init__(self, key: str, content: str, parent=None, metadata=None):
        """
        Initialise a skeleton node with the given parameters.

        Args:
            key (str): The identifier for this node (e.g., "1.", "1.1.")
            content (str): The textual content of this point
            parent: Optional parent SkeletonNode
            metadata: Optional dictionary of additional information
        """
        self.key = key  # The node identifier (e.g., "1.", "1.1.")
        self.content = content  # The textual content
        self.parent = parent  # Reference to parent node
        self.children = []  # Child nodes (subpoints)
        self.metadata = metadata or {}  # Additional information about this node

class SkeletonOfThoughts:
    """
    An enhanced implementation of the Skeleton of Thoughts approach.

    This class manages the creation, expansion, and linking of ideas
    in a hierarchical skeleton structure using a local Llama model.
    """

    def __init__(self, model_path, n_ctx=2048, n_threads=8, n_gpu_layers=35, temperature=0.7, max_retries=3, retry_delay=2):
        """
        Initialise the SkeletonOfThoughts with configuration parameters.

        Args:
            model_path (str): Path to the Llama model
            n_ctx (int): Context length for the model
            n_threads (int): Number of threads for multithreading
            n_gpu_layers (int): Number of layers to offload to GPU
            temperature (float): Sampling temperature for text generation
            max_retries (int): Maximum number of API call retries
            retry_delay (int): Delay in seconds between retries
        """
        self.client = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers
        )
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.skeleton = {}  # Dictionary representation of the skeleton
        self.root_nodes = []  # Root nodes in the tree structure
        self.all_nodes = {}  # All nodes indexed by key
        self.api_calls = 0  # Counter for API calls (for efficiency tracking)
        self.total_tokens = 0  # Counter for token usage

    @memoize(timeout=3600)  # Cache responses for 1 hour
    def execute_prompt(self, prompt, system_prompt=None, max_tokens=500):
        """
        Execute a prompt using the local Llama model with memoisation.

        This method is memoized to avoid repeated identical API calls,
        significantly improving performance for repeated or similar prompts.

        Args:
            prompt (str): The prompt to send to the model
            system_prompt (str): Optional custom system prompt
            max_tokens (int): Maximum number of tokens in the response

        Returns:
            str: The model response content
        """
        if system_prompt is None:
            system_prompt = "You are an AI assistant creating a detailed structured outline."

        # Increment API call counter
        self.api_calls += 1

        # Implement retry logic for robustness
        attempts = 0
        while attempts < self.max_retries:
            try:
                # Ajusta los argumentos según la documentación de llama_cpp
                response = self.client(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=self.temperature
                )

                # Track token usage
                self.total_tokens += response['usage']['total_tokens']

                return response['choices'][0]['text'].strip()

            except Exception as e:
                attempts += 1
                if attempts < self.max_retries:
                    print(f"API call failed, retrying in {self.retry_delay} seconds. Error: {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"API call failed after {self.max_retries} attempts: {str(e)}")
                    return ""

    def create_skeleton(self, topic, depth=2, points_per_level=3):
        """
        Create a hierarchical skeletal outline for the given topic.

        Args:
            topic (str): The topic to create a skeleton for
            depth (int): The depth of the hierarchy to generate
            points_per_level (int): Number of points to generate at each level

        Returns:
            dict: The created skeleton outline
        """
        # Clear any existing skeleton
        self.skeleton = {}
        self.root_nodes = []
        self.all_nodes = {}

        # Create a detailed prompt to guide the skeleton generation
        prompt = f"""Create a detailed skeletal outline for the topic: '{topic}'.

        Use the following hierarchical numbering format:
        1. First main point
          1.1. First subpoint of first main point
            1.1.1. First sub-subpoint (if depth allows)
          1.2. Second subpoint of first main point
        2. Second main point
          2.1. First subpoint of second main point

        Requirements:
        - Provide {points_per_level} main points
        - For each main point, provide {points_per_level} subpoints
        - {f"For each subpoint, provide {points_per_level} sub-subpoints" if depth > 2 else ""}
        - Each point should be clear, concise, and directly related to the topic
        - Points should be logical and follow a coherent structure
        - Each point should build on the previous one
        """

        try:
            # Get the raw skeleton text
            response = self.execute_prompt(
                prompt,
                system_prompt="You are an expert in creating detailed, well-structured outlines for complex topics.",
                max_tokens=800
            )

            # Parse the raw text into our structured format
            self.skeleton = self.parse_skeleton(response)

            # Build the tree structure
            self._build_tree_structure()

            return self.skeleton

        except Exception as e:
            print(f"Error creating skeleton: {str(e)}")
            return {}

    def parse_skeleton(self, text):
        """
        Parse the skeleton text into a structured dictionary and tree.

        This enhanced parser handles multi-level hierarchies and preserves
        the structure of the outline.

        Args:
            text (str): The raw skeleton text to parse

        Returns:
            dict: The parsed skeleton structure
        """
        # Split the text into lines and initialize variables
        lines = text.strip().split('\n')
        skeleton = {}
        current_keys = {}  # Track current key at each level

        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to match a numbered point at any level
            # This regex matches patterns like "1.", "1.1.", "1.1.1.", etc.
            match = re.match(r'^(\d+(?:\.\d+)*)\.\s*(.*)', line)

            if match:
                key = match.group(1) + "."
                content = match.group(2).strip()

                # Store in our dictionary
                skeleton[key] = content

                # Determine the depth level
                depth = key.count('.')
                current_keys[depth] = key

            # If it's not a numbered point, append to the current point
            elif current_keys:
                # Get the most recent key
                max_depth = max(current_keys.keys())
                current_key = current_keys[max_depth]

                # Append this line to the current point's content
                if current_key in skeleton:
                    skeleton[current_key] += " " + line

        return skeleton

    def _build_tree_structure(self):
        """
        Convert the flat dictionary structure into a hierarchical tree.

        This method creates SkeletonNode objects and establishes parent-child
        relationships based on the numbering system.
        """
        # Clear existing tree
        self.root_nodes = []
        self.all_nodes = {}

        # Sort keys to ensure proper processing order
        sorted_keys = sorted(self.skeleton.keys(),
                            key=lambda k: [int(n) for n in k.strip('.').split('.')])

        # First pass: Create all nodes
        for key in sorted_keys:
            node = SkeletonNode(key, self.skeleton[key])
            self.all_nodes[key] = node

            # If it's a top-level node (e.g., "1.", "2.")
            if key.count('.') == 1:
                self.root_nodes.append(node)

        # Second pass: Establish parent-child relationships
        for key, node in self.all_nodes.items():
            # Find parent key
            key_parts = key.strip('.').split('.')
            if len(key_parts) > 1:
                # Construct parent key by removing the last number
                parent_key = '.'.join(key_parts[:-1]) + '.'

                if parent_key in self.all_nodes:
                    parent = self.all_nodes[parent_key]
                    node.parent = parent
                    parent.children.append(node)

    def expand_point(self, point_key, depth=1):
        """
        Expand a specific point in the skeleton with more detail.

        Args:
            point_key (str): The key of the point to expand (e.g., "1.", "1.1.")
            depth (int): The depth of expansion (level of detail)

        Returns:
            str: The expanded point content
        """
        if point_key not in self.skeleton:
            return f"Point {point_key} not found in the skeleton."

        current_content = self.skeleton[point_key]

        # Create a context-aware prompt by including parent and sibling points
        context = self._get_point_context(point_key)
        context_text = "\n".join([f"{k}: {v}" for k, v in context.items()])

        # Create a detailed prompt for expansion
        detail_level = ["moderate", "significant", "comprehensive", "exhaustive"][min(depth-1, 3)]

        prompt = f"""Provide a {detail_level} expansion of this outline point:

        Point to expand: {point_key} {current_content}

        Context (surrounding points in the outline):
        {context_text}

        Expand this point by:
        1. Elaborating on the key concepts
        2. Providing specific examples or evidence
        3. Explaining the significance or implications
        4. Addressing potential counterarguments or limitations
        5. Making connections to broader themes in the outline

        Your expansion should be well-structured, detailed, and directly relevant to both the specific point and the overall topic.
        """

        # Get the expansion
        expansion = self.execute_prompt(prompt, max_tokens=500 * depth)

        # Update the skeleton with the expanded content
        expanded_content = f"{current_content}\n\nExpanded:\n{expansion}"
        self.skeleton[point_key] = expanded_content

        # Update the node content as well
        if point_key in self.all_nodes:
            self.all_nodes[point_key].content = expanded_content
            self.all_nodes[point_key].metadata["expanded"] = True
            self.all_nodes[point_key].metadata["expansion_depth"] = depth

        return expanded_content

    def _get_point_context(self, point_key):
        """
        Get the surrounding context of a point including parent, siblings, and children.

        Args:
            point_key (str): The key of the point

        Returns:
            dict: A dictionary of related points and their content
        """
        context = {}

        # If the point doesn't exist, return empty context
        if point_key not in self.all_nodes:
            return context

        node = self.all_nodes[point_key]

        # Add parent if it exists
        if node.parent:
            context[node.parent.key] = node.parent.content

        # Add siblings (excluding self)
        if node.parent:
            for sibling in node.parent.children:
                if sibling.key != point_key:
                    context[sibling.key] = sibling.content

        # Add children
        for child in node.children:
            context[child.key] = child.content

        return context

    def add_cross_link(self, point1_key, point2_key, bidirectional=True):
        """
        Add a cross-link between two points in the skeleton.

        Args:
            point1_key (str): The key of the first point
            point2_key (str): The key of the second point
            bidirectional (bool): Whether to add the link in both directions

        Returns:
            str: The created cross-link explanation
        """
        # Validate both points exist
        if point1_key not in self.skeleton or point2_key not in self.skeleton:
            missing = []
            if point1_key not in self.skeleton:
                missing.append(point1_key)
            if point2_key not in self.skeleton:
                missing.append(point2_key)
            return f"Points not found in the skeleton: {', '.join(missing)}"

        # Create an intelligent prompt for generating the relationship
        prompt = f"""Analyze the relationship between these two points in an outline:

        Point {point1_key}: {self.skeleton[point1_key]}

        Point {point2_key}: {self.skeleton[point2_key]}

        Provide:
        1. A concise explanation of how these points are related
        2. Key similarities or differences between them
        3. How they might complement or build upon each other
        4. How understanding one point enhances understanding of the other
        5. A short title that describes their relationship (in 3-5 words)

        Format your response as:
        "Link Title: [Your Title]

        [Your detailed explanation]"
        """

        # Generate the link content
        link_content = self.execute_prompt(prompt)

        # Extract the title if present
        title_match = re.match(r'Link Title:\s*(.+)', link_content)
        if title_match:
            link_title = title_match.group(1).strip()
        else:
            link_title = f"Link: {point1_key} - {point2_key}"

        # Create link key
        link_key = f"Link: {point1_key} -> {point2_key}"

        # Store the link in the skeleton
        self.skeleton[link_key] = link_content

        # Add the link as metadata to both nodes
        if point1_key in self.all_nodes and point2_key in self.all_nodes:
            node1 = self.all_nodes[point1_key]
            node2 = self.all_nodes[point2_key]

            # Initialize links list if not present
            if "links" not in node1.metadata:
                node1.metadata["links"] = []
            node1.metadata["links"].append({"to": point2_key, "content": link_content, "title": link_title})

            # If bidirectional, add link in reverse direction too
            if bidirectional:
                if "links" not in node2.metadata:
                    node2.metadata["links"] = []
                node2.metadata["links"].append({"to": point1_key, "content": link_content, "title": link_title})

                # Add reverse link to skeleton
                reverse_link_key = f"Link: {point2_key} -> {point1_key}"
                self.skeleton[reverse_link_key] = link_content

        return link_content

    def find_point_key(self, point_number):
        """
        Find the key in the skeleton dictionary for a given point number.

        Args:
            point_number (str or int): The point number to find (e.g., 1, "1.2")

        Returns:
            str: The key in the skeleton dictionary, or None if not found
        """
        # Convert to string if it's an integer
        point_str = str(point_number)

        # If it's already in the proper format with trailing dot
        if point_str.endswith('.') and point_str in self.skeleton:
            return point_str

        # Add trailing dot if not present
        if not point_str.endswith('.'):
            point_str += '.'

        # Check if it exists
        if point_str in self.skeleton:
            return point_str

        # If still not found, try pattern matching
        for key in self.skeleton.keys():
            if key.startswith(point_str):
                return key

        return None

    def generate_summary(self):
        """
        Generate a summary of the entire skeleton.

        Returns:
            str: A concise summary of the main points and their relationships
        """
        # Extract main points
        main_points = {k: v for k, v in self.skeleton.items()
                      if k.count('.') == 1 and not k.startswith('Link:')}

        if not main_points:
            return "No skeleton exists to summarize."

        # Create a prompt for the summary
        main_points_text = "\n".join([f"{k} {v}" for k, v in main_points.items()])

        prompt = f"""Create a concise summary of this outline, highlighting the main themes and relationships between points:

        {main_points_text}

        Your summary should:
        1. Identify the overarching theme
        2. Summarize the key points in 1-2 sentences each
        3. Describe how the points relate to each other
        4. Highlight any gaps or areas that could be developed further

        Keep your response under 300 words.
        """

        return self.execute_prompt(prompt, max_tokens=400)

    def optimize_structure(self):
        """
        Analyze and optimize the current skeleton structure.

        This method identifies and fixes issues such as:
        - Redundant or overlapping points
        - Logical gaps or missing connections
        - Imbalanced development across sections

        Returns:
            dict: The optimized skeleton
        """
        skeleton_text = "\n".join([f"{k} {v}" for k, v in self.skeleton.items()
                                  if not k.startswith('Link:')])

        prompt = f"""Analyze this outline for structural issues and suggest improvements:

        {skeleton_text}

        Identify and fix these issues:
        1. Redundant or overlapping points
        2. Logical gaps or missing connections
        3. Imbalanced development across sections
        4. Points that could be better organized or sequenced

        Return the entire improved outline using the same numbering system.
        """

        improved_skeleton_text = self.execute_prompt(prompt, max_tokens=800)

        # Parse the improved skeleton
        improved_skeleton = self.parse_skeleton(improved_skeleton_text)

        # Update the current skeleton
        self.skeleton = improved_skeleton
        self._build_tree_structure()

        return self.skeleton

    def display_skeleton(self, include_links=True, include_expanded=True):
        """
        Display the current skeleton structure in a hierarchical format.

        Args:
            include_links (bool): Whether to include cross-links
            include_expanded (bool): Whether to include expanded content
        """
        if not self.skeleton:
            print("The skeleton is empty.")
            return

        # Sort keys for proper display order
        def sort_key(k):
            if k.startswith('Link:'):
                return ('Z', k)  # Put links at the end
            return ('A', k)

        sorted_keys = sorted(self.skeleton.keys(), key=lambda k: sort_key(k))

        for key in sorted_keys:
            # Skip links if not included
            if not include_links and key.startswith('Link:'):
                continue

            value = self.skeleton[key]

            # Format expanded content
            if not include_expanded and 'Expanded:' in value:
                value = value.split('Expanded:')[0].strip()

            # Determine indentation level
            if key.startswith('Link:'):
                indent = '🔄 '
            else:
                # Calculate indentation based on key depth
                depth = key.count('.') - 1
                indent = '  ' * depth + ('📌 ' if depth == 0 else '• ')

            # Print the formatted point
            print(f"{indent}{key} {value}\n")

    def export_to_markdown(self, filename=None):
        """
        Export the skeleton to a Markdown file.

        Args:
            filename (str): Optional filename to save to

        Returns:
            str: The markdown content
        """
        lines = []
        lines.append("# Skeleton of Thoughts\n")

        # Sort keys for proper hierarchy
        sorted_keys = sorted(self.skeleton.keys(),
                            key=lambda k: 'Z' + k if k.startswith('Link:') else k)

        # First add all points
        for key in sorted_keys:
            if not key.startswith('Link:'):
                value = self.skeleton[key]
                depth = key.count('.') - 1

                # Handle expanded content specially
                if 'Expanded:' in value:
                    basic, expanded = value.split('Expanded:', 1)
                    header = '#' * (depth + 2)
                    lines.append(f"{header} {key} {basic.strip()}")
                    lines.append(f"{expanded.strip()}")
                else:
                    header = '#' * (depth + 2)
                    lines.append(f"{header} {key} {value}")

                lines.append("")

        # Then add links in a separate section
        has_links = any(k.startswith('Link:') for k in sorted_keys)
        if has_links:
            lines.append("\n## Cross-Links\n")

            for key in sorted_keys:
                if key.startswith('Link:'):
                    points = key.replace('Link:', '').strip()
                    lines.append(f"### {points}")
                    lines.append(self.skeleton[key])
                    lines.append("")

        # Join lines into markdown content
        markdown = "\n".join(lines)

        # Write to file if filename is provided
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"Skeleton exported to {filename}")

        return markdown

    def performance_report(self):
        """
        Generate a report on performance metrics.

        Returns:
            dict: Performance statistics
        """
        execution_time = time.time() - start_time

        report = {
            "execution_time": execution_time,
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.total_tokens * 0.002 / 1000,  # Approximate cost
            "skeleton_size": len(self.skeleton),
            "cache_hits": getattr(self.execute_prompt, 'cache_size', lambda: 0)(),
        }

        if self.api_calls > 0:
            report["avg_time_per_call"] = execution_time / self.api_calls
            report["avg_tokens_per_call"] = self.total_tokens / self.api_calls

        return report

def main():
    """
    Main function to demonstrate the usage of the enhanced SkeletonOfThoughts class.
    """
    model_path = "../Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # Replace with your actual model path

    try:
        # Create an instance with optimal settings
        sot = SkeletonOfThoughts(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=35,
            temperature=0.7,
            max_retries=3
        )

        # Define the topic and create the skeleton
        topic = "The Impact of Artificial Intelligence on Future Job Markets"
        print(f"Creating skeleton for topic: {topic}")
        sot.create_skeleton(topic, depth=2, points_per_level=3)

        print("\nInitial Skeleton:")
        sot.display_skeleton(include_expanded=False)

        # Expand a few points
        print("\nExpanding Point 1:")
        sot.expand_point("1.", depth=2)

        print("\nExpanding Point 2.1:")
        sot.expand_point("2.1.", depth=1)

        # Add cross-links
        print("\nAdding Cross-links:")
        sot.add_cross_link("1.", "2.")
        sot.add_cross_link("1.1.", "2.1.")

        # Generate a summary
        print("\nSkeleton Summary:")
        summary = sot.generate_summary()
        print(summary)

        # Display the final skeleton
        print("\nFinal Skeleton:")
        sot.display_skeleton()

        # Export to markdown
        markdown = sot.export_to_markdown("skeleton_output.md")

        # Show performance metrics
        print("\nPerformance Report:")
        report = sot.performance_report()
        for key, value in report.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
