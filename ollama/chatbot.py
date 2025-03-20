import ollama
import subprocess
import time
import customtkinter as ctk
import threading
import subprocess
import time
import ollama
import re

class BaseAgent:
    def __init__(self,system_prompt,few_shots,environment_definitions, model="llama3:8b"):
        self.model = model
        self.process = None  
        self.system_prompt=system_prompt
        self.environment_definitions=environment_definitions
        self.few_shots=few_shots
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



class EffectAgent(BaseAgent):
    def __init__(self,system_prompt,few_shots,environment_definitions, model="llama3:8b"):
        super().__init__(system_prompt,few_shots,environment_definitions, model=model)

    def generate_effect(self, user_input):
        prompt = f"""
{self.system_prompt}

### Output Instruction:
**Ensure the output is just the effect function. Do not return any explanatory text at all.**

### Few-Shot Examples:

{self.few_shots}
### Now generate an effect function based on the following advice:

Advice = "{user_input}"
Primitives = {self.primitives}
Effect =
"""
        
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        cleaned_output = self.clean_effect_output(response['message']['content'].strip())
        
        return cleaned_output
     

    def clean_effect_output(self, raw_output):
        """Removes anything before 'Effect main' to ensure clean formatting."""
        effect_start = raw_output.find("Effect main")
        if effect_start != -1:
            return raw_output[effect_start:]  
        return raw_output  
class PolicyAgent(BaseAgent):
    def __init__(self,system_prompt,few_shots,environment_definitions, model="llama3:8b"):
        super().__init__(system_prompt,few_shots,environment_definitions, model=model)

    def generate_policy(self, user_input):
        """Generates a structured taxi policy using Ollama based on user input."""
        
        prompt = f"""
{self.system_prompt}

### Output Instruction:
 **Ensure the output is just the policy. Do not return any explanatory text at all.**

### Few-Shot Examples:
{self.few_shots}

### Now generate a policy based on the following advice:

Advice = "{user_input}"
Primitives = ['move_n', 'move_s', 'move_e', 'move_w',passenger_x,passenger_y,x,y,carrying_passenger]
Policy =
"""
        
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])

        cleaned_output = self.clean_policy_output(response['message']['content'].strip())
        
        return cleaned_output

    def clean_policy_output(self, raw_output):
        """Removes anything before 'Policy <dynamic_name>:' to ensure clean formatting."""
        match = re.search(r"(?m)^Policy \w+:", raw_output)  # Match any line that starts with "Policy <name>:"
        if match:
            return raw_output[match.start():]  # Keep everything from "Policy <name>:" onwards
        return raw_output  # If no match is found, return the original response

class ChatApp(ctk.CTk):
    def __init__(self, effect_agent, policy_agent,environment_constants):
        super().__init__()
        self.effect_agent = effect_agent
        self.policy_agent = policy_agent
        self.effect_text = None
        self.policy_text = None
        self.environment_constants=environment_constants

        self.title("Ollama Taxi RLang Policy Generator")
        self.attributes("-fullscreen", True)  

        ctk.set_appearance_mode("dark") 

        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.chat_frame = ctk.CTkScrollableFrame(self, width=1200, height=700)
        self.chat_frame.grid(row=0, column=0, padx=20, pady=20, columnspan=2, sticky="nsew")

        self.user_input = ctk.CTkEntry(self, width=1000, placeholder_text="Describe your policy advice...")
        self.user_input.grid(row=1, column=0, padx=20, pady=20, sticky="ew")

        self.generate_button = ctk.CTkButton(self, text="Generate Effect", command=self.generate_output)
        self.generate_button.grid(row=1, column=1, padx=10, pady=20)

        self.toggle_button = ctk.CTkButton(self, text="Switch to Policy", command=self.toggle_mode)
        self.toggle_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.compile_button = ctk.CTkButton(self, text="Compile to File", command=self.compile_to_file)
        self.compile_button.grid(row=2, column=1, padx=20, pady=10)

        self.user_input.bind("<Return>", lambda event: self.generate_output())

        self.is_generating_effect = True

    def add_chat_bubble(self, text, sender="user"):
        bubble_color = "#1E88E5" if sender == "user" else "#2E2E2E"
        text_align = "e" if sender == "user" else "w"
        bubble = ctk.CTkLabel(self.chat_frame, text=text, wraplength=900, justify="left",
                              fg_color=bubble_color, text_color="white", corner_radius=10, padx=15, pady=10)
        bubble.pack(anchor=text_align, pady=5, padx=20)
        self.chat_frame._parent_canvas.yview_moveto(1)

    def generate_output(self):
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        
        self.add_chat_bubble(f"You: {user_text}", sender="user")

        typing_bubble = ctk.CTkLabel(self.chat_frame, text="Generating...", wraplength=900, justify="left",
                                    fg_color="#2E2E2E", text_color="white", corner_radius=10, padx=15, pady=10)
        typing_bubble.pack(anchor="w", pady=5, padx=20)

        self.update_idletasks()

        def fetch_output():
            if self.is_generating_effect:
                response = self.effect_agent.generate_effect(user_text)
                self.effect_text = response
            else:
                response = self.policy_agent.generate_policy(user_text)
                self.policy_text = response
            typing_bubble.destroy()

            self.add_chat_bubble(response, sender="bot")

        threading.Thread(target=fetch_output, daemon=True).start()
        self.user_input.delete(0, "end")
        
    def toggle_mode(self):
        """Toggles between Effect and Policy generation."""
        self.is_generating_effect = not self.is_generating_effect
        if self.is_generating_effect:
            self.generate_button.configure(text="Generate Effect")
            self.toggle_button.configure(text="Switch to Policy")
        else:
            self.generate_button.configure(text="Generate Policy")
            self.toggle_button.configure(text="Switch to Effect")

    def compile_to_file(self):
        """Compiles the generated effect and policy into a text file."""
        if not self.effect_text or not self.policy_text:
            self.add_chat_bubble("Both Effect and Policy must be generated before compiling!", sender="bot")
            return
        
        file_content = f"\n{self.environment_constants}\n{self.effect_text}\n\n{self.policy_text}"
        file_path = "compiled_policy.rlang"

        with open(file_path, "w") as file:
            file.write(file_content)

        self.add_chat_bubble(f"Compiled successfully! Saved to {file_path}", sender="bot")
import constants
if __name__ == "__main__":
    stage1 = EffectAgent(system_prompt=constants.effect_prompt, few_shots=constants.effect_fewshots, environment_definitions=constants.environment_definitions)
    print(stage1)
    stage2 = PolicyAgent(system_prompt=constants.policy_prompt, few_shots=constants.policy_fewshots, environment_definitions=constants.environment_definitions)

    try:
        stage1.start_ollama_serve()
        app = ChatApp(stage1, stage2,environment_constants=constants.environment_definitions)
        app.mainloop()
    finally:
        stage1.stop_ollama_serve()