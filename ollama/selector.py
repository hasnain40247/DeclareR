import os
import sys
import json
import gymnasium as gym
from datetime import datetime
import importlib.util
import rlang


# Maps algorithm names to folder/module names
AGENT_MODULES = {
    "Q-Learning": "q_learning",
    "Dyna-Q": "dyna_q",
    "R-Max": "r_max"
}


class RLangValidator:
    def __init__(self, env_name, rlang_file, policy_name=None):
        self.env_name = env_name.lower()
        self.rlang_file = rlang_file
        self.policy_name = None if policy_name == "0" else policy_name
        self.env = self._make_env()
        self.knowledge = None
        self.log_lines = []

    def _make_env(self):
        if self.env_name == "taxi":
            return gym.make("Taxi-v3")
        return gym.make("CliffWalking-v0")

    def log(self, *args, print_to_console=False):
        message = "[LOG] " + " ".join(str(arg) for arg in args)
        self.log_lines.append(message)
        if print_to_console:
            print(message)

    def write_log_file(self, prefix="validation"):
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./logs/{prefix}_{timestamp}.log"
        with open(filename, "w") as f:
            f.write("\n".join(self.log_lines))
        # print(f"[LOG] Saved log to {filename}")

    def run(self, algorithm,hyperparameters):
        module_folder = AGENT_MODULES.get(algorithm)
    
        base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
     
        module_path = os.path.join(base_dir, self.env_name, f"{module_folder}.py")

        if not os.path.isfile(module_path):
            self.log(f"Agent file not found: {module_path}")
            return

       

        module_name = f"{self.env_name}_{module_folder}"
      
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)


        class_name = f"RLang{algorithm.replace('-', '')}Agent"
      
        AgentClass = getattr(agent_module, class_name)

        
        def parse_val(val):
            if isinstance(val, str) and val.lower() == "none":
                return None
            return val
    
  

     
    
        parsed_params = {k: parse_val(v) for k, v in hyperparams.items()}
        self.knowledge = parsed_params.pop("knowledge", None)


        if self.knowledge is not None and os.path.exists(self.knowledge):
            file=self.knowledge.split("/")[-1]
            self.knowledge = rlang.parse_file(self.knowledge.split("/")[-1])

            if algorithm=="Dyna-Q" :
                policy_name=extract_policy_name(file)

                parsed_params["policy_name"]=policy_name
                self.log("Policy loaded from:", file)

     
            self.log("Knowledge loaded from:", file)
     
        agent = AgentClass(self.env, knowledge=self.knowledge, **parsed_params)
        if algorithm=="R-Max":
            rewards = agent.train(episodes=100)
        else:
            rewards = agent.train(episodes=15000)
            # rewards = agent.train(episodes=10)



        self.log(f"{algorithm} training complete. Episodes: {len(rewards)}")
        self.write_log_file()

import re
def extract_policy_name(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Regular expression to find the policy name
    policy_name_match = re.search(r'Policy\s+(\w+):', content)
    
    if policy_name_match:
        return policy_name_match.group(1)  # Return the policy name found
    
    return None  # If no policy is found


if __name__ == "__main__":


    _, rlang_file, env_name, algorithm, hyperparams_json = sys.argv
 

    rlang_file = os.path.basename(rlang_file)
    rlang_path = os.path.join(os.getcwd(), rlang_file)
   
 
    
    try:
        hyperparams = json.loads(hyperparams_json)
    except json.JSONDecodeError:
        # print("[ERROR] Could not parse hyperparams JSON.")
        sys.exit(1)

  


    # Initialize and run validator
    validator = RLangValidator(env_name, rlang_path)
    validator.run(algorithm,hyperparams)
