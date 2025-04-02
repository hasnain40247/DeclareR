import ollama
from BaseAgent import BaseAgent
import re
class PolicyAgent(BaseAgent):
    def __init__(self,system_prompt,few_shots,environment_definitions,vocab, model="gemma3:12b"):
        super().__init__(system_prompt,few_shots,environment_definitions,vocab=vocab, model=model)

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
Primitives = {self.primitives}
Policy =
"""
        
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])

        cleaned_output = self.clean_policy_output(response['message']['content'].strip())
        
        return cleaned_output
    
    def generate_policy_stream(self, user_input):
            prompt = f"""
            {self.system_prompt}

            ### Output Instruction:
            **Ensure the output is just the policy. Do not return any explanatory text at all.**

            ### Few-Shot Examples:
            {self.few_shots}

            ### Now generate a policy based on the following advice:

            Advice = "{user_input}"
            Primitives = {self.primitives}
            Policy =
            """
            stream = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], stream=True)
            return stream 
    def refine_policy_stream(self, original_Policy, refinement_instruction):
            prompt = f"""
        {self.system_prompt}

        ### Rules:
            - Take the provided `Policy` function.
            - Apply the user’s feedback **exactly**.
            - If prompted to rename the policy replace the current name with the new one.
            - Strictly return the **revised** `Policy` only — do NOT explain your changes.
            - Do NOT include any headings like 'Policy =' or additional comments.
   

            

        **Original Policy:**
        {original_Policy}

        **Refinement Instruction:**
        "{refinement_instruction}"

        Policy =
        """
            stream = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], stream=True)
            return stream

    def clean_policy_output(self, raw_output):
        """Removes anything before 'Policy <dynamic_name>:' to ensure clean formatting."""
        match = re.search(r"(?m)^Policy \w+:", raw_output)  # Match any line that starts with "Policy <name>:"
        if match:
            return raw_output[match.start():]  # Keep everything from "Policy <name>:" onwards
        return raw_output  # If no match is found, return the original response
