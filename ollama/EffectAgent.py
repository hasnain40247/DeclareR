import ollama
from BaseAgent import BaseAgent


# modesl tried
# llama3:8b
#


class EffectAgent(BaseAgent):
    def __init__(self,system_prompt,few_shots,environment_definitions,vocab=None, model="gemma3:12b"):
        super().__init__(system_prompt,few_shots,environment_definitions,vocab=vocab, model=model)

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
    def generate_effect_stream(self, user_input):
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
        stream = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], stream=True)
        return stream 

    def refine_effect_stream(self, original_effect, refinement_instruction):
        prompt = f"""
    {self.system_prompt}

    ### Rules:
        - Take the provided `Effect` function.
        - Apply the user’s feedback **exactly**.
        - Strictly return the **revised** `Effect` only — do NOT explain your changes.
        - Do NOT include any headings like 'Effect =' or additional comments.

    **Original Effect:**
    {original_effect}

    **Refinement Instruction:**
    "{refinement_instruction}"

    Effect =
    """
        stream = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], stream=True)
        return stream


    def clean_effect_output(self, raw_output):
        """Removes anything before 'Effect main' to ensure clean formatting."""
        effect_start = raw_output.find("Effect main")
        if effect_start != -1:
            return raw_output[effect_start:]  
        return raw_output  