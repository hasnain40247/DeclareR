import ollama
from BaseAgent import BaseAgent
import re
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
