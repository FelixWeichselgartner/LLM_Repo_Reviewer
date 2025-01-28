import os
import json
import logging
from ollama import chat
from ollama import ChatResponse

# Configure logging
LOG_FILE = "llm_interaction.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

MAX_LINES = 1000
MAX_BYTES = 100 * 1024  # 100 KB

EXCLUDED_EXTENSIONS = {".pyc", ".tar", '.7z', ".zip", ".mp4", ".mp3", ".pdf", '.jpg', '.png', 'tar.lz'}
EXCLUDED_FILES = {"license", "license.md", 'license.txt', '.gitignore'}
MAX_DEPTH = 2

def load_gitignore(root_path):
    """Loads and parses the .gitignore file to get a list of ignored patterns."""
    gitignore_path = os.path.join(root_path, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as file:
            return [line.strip() for line in file if line.strip() and not line.startswith("#")]
    return []

def is_ignored(file_path, ignored_patterns):
    """Checks if a file or folder matches any ignored pattern or exclusion criteria."""
    if any(file_path.endswith(ext.lower()) for ext in EXCLUDED_EXTENSIONS):
        return True
    if os.path.basename(file_path).lower() in EXCLUDED_FILES:
        return True
    return any(file_path.startswith(pattern) or pattern in file_path.split(os.sep) for pattern in ignored_patterns)

def get_depth(relative_path):
    """Calculates the depth of a relative path."""
    return len(relative_path.split(os.sep))

def generate_file_tree_and_content(repo_path):
    """Generates a file tree and reads file contents, excluding ignored files and large files."""
    ignored_patterns = load_gitignore(repo_path)
    repo_data = {}

    for root, dirs, files in os.walk(repo_path):
        # Exclude .git folder
        if ".git" in dirs:
            dirs.remove(".git")

        # Calculate depth and exclude deeper levels
        relative_path = os.path.relpath(root, repo_path)
        if relative_path == ".":
            relative_path = ""
        if get_depth(relative_path) >= MAX_DEPTH:
            continue

        # Remove ignored directories from traversal
        dirs[:] = [d for d in dirs if not is_ignored(os.path.relpath(os.path.join(root, d), repo_path), ignored_patterns)]

        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_file_path = os.path.join(relative_path, file_name)

            if is_ignored(relative_file_path, ignored_patterns):
                continue

            try:
                if os.path.getsize(file_path) > MAX_BYTES:
                    logging.info(f"Skipping large file: {relative_file_path}")
                    continue

                with open(file_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > MAX_LINES:
                        logging.info(f"Skipping file with too many lines: {relative_file_path}")
                        continue
                    repo_data[relative_file_path] = "".join(lines)
            except Exception as e:
                logging.error(f"Error reading file {relative_file_path}: {str(e)}")

    return repo_data

def send_message_to_llm(messages):
    """Sends a message to the LLM and logs the interaction."""
    try:
        logging.info(f"Sending message to LLM: {messages[-1]['content']}")
        # phi4
        response: ChatResponse = chat(model='deepseek-r1:1.5b', messages=messages)
        logging.info(f"Received response from LLM: {response.message.content}")
        return response.message.content
    except Exception as e:
        logging.error(f"Error during LLM interaction: {str(e)}")
        return f"Error: {str(e)}"

def handle_repos(repo_paths):
    for root_path in repo_paths:
        """Processes each repository and sends its data to the LLM for evaluation."""
        repos = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        
        for repo in repos:
            repo_path = os.path.join(root_path, repo)
            logging.info(f"Processing repository: {repo}")

            # Generate file tree and contents for the repository
            repo_data = generate_file_tree_and_content(repo_path)

            messages = [
                {
                    'role': 'user',
                    'content': (
                        f"Repository: {repo}\n\n" +
                        "Here is the file tree and contents of the repository:\n\n" +
                        json.dumps(repo_data, indent=2) +
                        "\n\nPlease evaluate this repository on the following criteria:\n" +
                        "- Monetary Potential\n" +
                        "- Uniqueness\n" +
                        "- Quality\n" +
                        "Provide a rating for each criterion and any additional insights you may have."
                    ),
                }
            ]

            llm_response = send_message_to_llm(messages)
            print(f"Response for {repo}:\n", llm_response)

def main():
    repo_paths = ["../"]

    # Process repositories and interact with the LLM
    handle_repos(repo_paths)

if __name__ == "__main__":
    main()
