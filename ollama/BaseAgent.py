import time
import re
import subprocess
import json
class BaseAgent:
    def __init__(self,system_prompt,few_shots,environment_definitions,vocab, model="gemma3:12b"):
        self.model = model
        self.process = None  
        self.system_prompt=system_prompt
        self.environment_definitions=environment_definitions
        self.few_shots=few_shots
        self.vocab_path = vocab

        self.primitives= self.extract_primitives()
        
    def extract_primitives(self):
        """Parses the environment definitions to extract a list of valid primitives."""
        primitives = set()

        # Extract Factors
        factor_pattern = re.compile(r"Factor (\w+) :=")
        primitives.update(factor_pattern.findall(self.environment_definitions))

        # Extract Propositions
        proposition_pattern = re.compile(r"Proposition (\w+) :=")
        primitives.update(proposition_pattern.findall(self.environment_definitions))

        # Extract Actions
        action_pattern = re.compile(r"Action (\w+) :=")
        primitives.update(action_pattern.findall(self.environment_definitions))

        if self.vocab_path:
            try:
                with open(self.vocab_path, "r") as f:
                    vocab_data = json.load(f)
                    features = vocab_data.get("vocabulary", {}).get("features", [])
                    for feature in features:
                        name = feature.get("name")
                        if name:
                            primitives.add(name)
            except Exception as e:
                print(f"[⚠️] Failed to parse vocab.json at {self.vocab_path}: {e}")

  
        return f"[{', '.join(sorted(primitives))}]"


    def start_ollama_serve(self):
        """Starts the Ollama server."""
        print("[✓] Starting Ollama server...")
        self.process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)  # Give it time to start

    def stop_ollama_serve(self):
        """Stops the Ollama server."""
        if self.process:
            print("[✓] Stopping Ollama server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("[✓] Ollama server stopped.")

