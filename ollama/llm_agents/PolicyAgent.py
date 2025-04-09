import ollama
from llm_agents.BaseAgent import BaseAgent
import re

class PolicyAgent(BaseAgent):
    def __init__(self, system_prompt, few_shots, environment_definitions, vocab=None, model="llama3:8b"):
        super().__init__(system_prompt, few_shots, environment_definitions, vocab=vocab, model=model)
        self.chat_history = []
        self.initial_prompt = self._build_intro_prompt()

    def _build_intro_prompt(self):
        """Initial setup message including system, few-shots, and task context."""
        return f"""
{self.system_prompt}

### Output Instruction:
**Ensure the output is just the policy. Do not return any explanatory text at all.**

### Few-Shot Examples:
{self.few_shots}

Now continue the conversation. I will give you natural language advice. You will respond with structured policy blocks only.
"""

    def generate_stream(self, user_input):
        """Handles both first policy and follow-up refinements with chat history."""
        # Set up chat history if first time
        if not self.chat_history:
            self.chat_history.append({"role": "user", "content": self.initial_prompt})
            self.chat_history.append({
                "role": "assistant",
                "content": "Understood. I'm ready for your first piece of advice."
            })

        formatted_advice = f'Advice = "{user_input}"\nPrimitives = {self.primitives}\nPolicy ='
        self.chat_history.append({"role": "user", "content": formatted_advice})

        stream = ollama.chat(model=self.model, messages=self.chat_history, stream=True)
        return stream

    def register_response(self, policy_response):
        """Add assistant policy response to chat history, cleaned."""
        cleaned = self.clean_policy_output(policy_response)
        self.chat_history.append({"role": "assistant", "content": cleaned})

    def clean_policy_output(self, raw_output):
        """Extract everything from 'Policy <name>:' onward."""
        match = re.search(r"(?m)^Policy \w+:", raw_output)
        if match:
            return raw_output[match.start():]
        return raw_output

    def reset_history(self):
        """Start a new policy conversation."""
        self.chat_history = []
