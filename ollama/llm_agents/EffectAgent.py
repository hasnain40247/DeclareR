# import ollama
# from ollama.llm_agents.BaseAgent import BaseAgent


# # modesl tried
# # llama3:8b
# # gemma3:12b


# class EffectAgent(BaseAgent):
#     def __init__(self,system_prompt,few_shots,environment_definitions,vocab=None, model="llama3:8b"):
#         super().__init__(system_prompt,few_shots,environment_definitions,vocab=vocab, model=model)

#     def generate_effect(self, user_input):
#         prompt = f"""
# {self.system_prompt}

# ### Output Instruction:
# **Ensure the output is just the effect function. Do not return any explanatory text at all.**

# ### Few-Shot Examples:

# {self.few_shots}
# ### Now generate an effect function based on the following advice:

# Advice = "{user_input}"
# Primitives = {self.primitives}
# Effect =
# """
        
#         response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
#         cleaned_output = self.clean_effect_output(response['message']['content'].strip())
        
#         return cleaned_output
#     def generate_effect_stream(self, user_input):
#         prompt = f"""
#     {self.system_prompt}

#     ### Output Instruction:
#     **Ensure the output is just the effect function. Do not return any explanatory text at all.**

#     ### Few-Shot Examples:

#     {self.few_shots}
#     ### Now generate an effect function based on the following advice:

#     Advice = "{user_input}"
#     Primitives = {self.primitives}
#     Effect =
#     """
#         stream = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], stream=True)
#         return stream 

#     def refine_effect_stream(self, original_effect, refinement_instruction):
#         prompt = f"""
#     {self.system_prompt}

#     ### Rules:
#         - Take the provided `Effect` function.
#         - Apply the user’s feedback **exactly**.
#         - Strictly return the **revised** `Effect` only — do NOT explain your changes.
#         - Do NOT include any headings like 'Effect =' or additional comments.

#     **Original Effect:**
#     {original_effect}

#     **Refinement Instruction:**
#     "{refinement_instruction}"

#     Effect =
#     """
#         stream = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], stream=True)
#         return stream


#     def clean_effect_output(self, raw_output):
#         """Removes anything before 'Effect main' to ensure clean formatting."""
#         effect_start = raw_output.find("Effect main")
#         if effect_start != -1:
#             return raw_output[effect_start:]  
#         return raw_output  

import ollama
from llm_agents.BaseAgent import BaseAgent

class EffectAgent(BaseAgent):
    def __init__(self, system_prompt, few_shots, environment_definitions, vocab=None, model="llama3:8b"):
        super().__init__(system_prompt, few_shots, environment_definitions, vocab=vocab, model=model)
        self.chat_history = []
        self.initial_prompt = self._build_intro_prompt()

    def _build_intro_prompt(self):
        """Builds system, few-shots, and environment setup as an initial user message."""
        return f"""
{self.system_prompt}

### Output Instruction:
**Ensure the output is just the effect function. Do not return any explanatory text at all.**

### Few-Shot Examples:

{self.few_shots}

Now continue the conversation. I will provide natural language advice, and you will reply with the correct Effect block.
"""

    def generate_stream(self, user_input):
        """Handles both initial and follow-up advice, using chat memory."""
        # If chat just started, initialize with the full setup
        if not self.chat_history:
            self.chat_history.append({"role": "user", "content": self.initial_prompt})
            self.chat_history.append({
                "role": "assistant",
                "content": "Understood. I'm ready for your first piece of advice."
            })

        # Add the user’s latest advice
        formatted_advice = f'Advice = "{user_input}"\nPrimitives = {self.primitives}\nEffect ='
        self.chat_history.append({"role": "user", "content": formatted_advice})

        # Call Ollama with the full history
        stream = ollama.chat(model=self.model, messages=self.chat_history, stream=True)
        return stream

    def register_response(self, effect_response):
        """Store the assistant's last output in chat history."""
        cleaned = self.clean_effect_output(effect_response)
        self.chat_history.append({"role": "assistant", "content": cleaned})

    def clean_effect_output(self, raw_output):
        """Strip out unnecessary prefixes or text before 'Effect main'."""
        effect_start = raw_output.find("Effect main")
        return raw_output[effect_start:] if effect_start != -1 else raw_output

    def reset_history(self):
        """Clear chat history to start fresh."""
        self.chat_history = []
