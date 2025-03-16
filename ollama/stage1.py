import ollama
import subprocess
import time
import os
class Stage1:
    def __init__(self, model="llama3:8b"):

        self.model = model

    def prompter(self, advice):
        prompt = f"""
            RLang is a formal language for specifying information about every element of a Markov
            Decision Process (S, A, R, T). Each RLang object refers to one or more elements of an MDP.
            Here is a description of three important RLang groundings:

            - **Policy**: A direct function from states to actions, best used for more general commands.
            - **Effect**: A prediction about the state of the world or the reward function.
            - **Plan**: A sequence of specific steps to take.

            Your task is to determine which **single** RLang grounding most naturally corresponds to a given piece of advice.  
            Respond only with one of the following three words: "Policy", "Effect", or "Plan" nothing else.

            **Examples:**  
            Advice = "If the passenger is in the taxi, go to the destination."
            Grounding: Policy

            Advice = "Dropping off the passenger at the wrong location will result in a penalty."
            Grounding: Effect

            Advice = "First pick up the passenger, then drive to their destination, and finally drop them off."
            Grounding: Plan

            Advice = "If there is a passenger at the current location, pick them up."
            Grounding: Policy

            Advice = "If you try to move into a wall, your action will have no effect."
            Grounding: Effect

            Advice = "Pick up the passenger at (2,3), drop them off at (4,1)."
            Grounding: Plan

            **Now classify the following advice:**  
            Advice = "{advice}"  
            """

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()


    
    def start_ollama_serve(self):

        print("[\u2713] Starting Ollama server.")
        process=subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)
        return process

    def stop_ollama_serve(self,process):
     
        print("[\u2713] Stopping Ollama server.")
        process.terminate()  
        try:
            process.wait(timeout=5)  
        except subprocess.TimeoutExpired:
            process.kill()  

if __name__ == "__main__":

    stage1 = Stage1()  
    ollama_process = stage1.start_ollama_serve()

    advice_samples = [
        "Don't eat food off the ground.",
        "If you see a red light, stop.",
        "Pick up the sword before fighting the dragon.",
        "First take the key, then unlock the door.",
        "If the enemy is near, run away."
    ]

    print("\n--- Testing RLang Grounding ---")
    for advice in advice_samples:
        grounding = stage1.prompter(advice)
        print(f"Advice: {advice}")
        print(f"Grounding: {grounding}\n")


    stage1.stop_ollama_serve(ollama_process)
