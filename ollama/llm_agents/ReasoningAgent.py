import ollama

import time
import subprocess

class ReasoningAgent():
    def __init__(self, model="llama3:8b"):
        self.model = model

    def generate_reasoning_stream(self, state, action, rgb_array=None):
        """
        Generate reasoning with streaming, allowing incremental responses.
        
        Args:
        - state (str): The current state of the environment.
        - action (str): The action taken by the agent.
        - rgb_array (np.array or list): The RGB array of the environment, can be a visual representation.
        
        Returns:
        - generator: A streaming generator that yields the reasoning chunks.
        """
        rgb_input = ""
        if rgb_array is not None:
            rgb_input = f"RGB Array (Visual Input): {str(rgb_array)}"

        prompt = f"""
            You are a NLP reasoning agent that, upon given a state, action, you reason as to why you have chosen that

            ### Output Instruction:
            **Ensure the output is just the reasoning behind the action. Do not return any explanatory text at all.**

            ### Now generate reasoning for the following action based on the current state and visual input:

            State: "{state}"
            Action: "{action}"
            {rgb_input}

            Reasoning:
            """
        
        # Use streaming for the response
        stream = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], stream=True)

        # Yield each chunk of the streamed response
        for chunk in stream:
            # Assuming `chunk['message']['content']` contains the reasoning text
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']  # Yield the content incrementally

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
