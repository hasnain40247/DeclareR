import ollama
import subprocess
import time

class ReasoningAgent:
    def __init__(self,fewshots, model="mistral:7b-instruct-v0.2-q8_0"):
        self.model = model
        self.process = None  
        self.few_shots=fewshots

    def build_prompt(self, state_action_description):
        """
        Build a prompt using the state-action description for a single step.
        """
        header = """You are an intelligent reasoning assistant that explains agent decisions in the OpenAI Taxi-v3 environment.

        In this environment:
        - The world is a 5x5 grid with four fixed landmarks:
            R (Red): (0, 0), G (Green): (0, 4), Y (Yellow): (4, 0), B (Blue): (4, 3)
        - A taxi must pick up a passenger and drop them off at their destination.
        - The state includes the taxi’s position, the passenger’s location (or whether they’re in the taxi), and the destination.
        - The agent can take one of the following actions:
            0 = South, 1 = North, 2 = East, 3 = West, 4 = Pickup, 5 = Dropoff

        Given a state and the action taken, explain the rationale behind the agent's choice and keep it brief.

        ### Output Instruction:
        **Ensure that only the rationale is returned. Do not return any explanatory text at all.**

        ### Few-Shot Examples:
        {self.few_shots}
        """
        
        prompt = f"{header}\n\n{state_action_description}\nRationale:\n"
        return prompt

    def generate_reasoning_stream(self, state_action_description):
        """
        Generate streamed rationale for a single state-action description.
        """
        prompt = self.build_prompt(state_action_description)
        stream = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], stream=True)

        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']

    def start_ollama_serve(self):
        """Starts the Ollama server (if needed)."""
        print("[✓] Stopping any existing Ollama server...")
        subprocess.call(["pkill", "-f", "ollama"])

        time.sleep(2)  # Small delay to ensure the process is stopped

        print("[✓] Starting Ollama server...")
        self.process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    

        time.sleep(5) 



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
