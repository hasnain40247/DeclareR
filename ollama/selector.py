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
    "R-Max": "rmax"
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

    def log(self, *args, print_to_console=True):
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
        print(f"[LOG] Saved log to {filename}")

    def run(self, algorithm,hyperparameters):
        module_folder = AGENT_MODULES.get(algorithm)
        # if not module_folder:
        #     self.log(f"Unsupported algorithm: {algorithm}")
        #     return

        # Base directory (e.g. /Users/hasnainsikora/Projects/DeclareR)
        base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
     
        module_path = os.path.join(base_dir, self.env_name, f"{module_folder}.py")

        if not os.path.isfile(module_path):
            self.log(f"Agent file not found: {module_path}")
            return

        # # Import the agent module dynamically
        print("module_path")
        print(module_path)
        print()

        module_name = f"{self.env_name}_{module_folder}"
      
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)

        # Class name (e.g., RLangQLearningAgent)
        class_name = f"RLang{algorithm.replace('-', '')}Agent"
      
        AgentClass = getattr(agent_module, class_name)

        
        def parse_val(val):
            if isinstance(val, str) and val.lower() == "none":
                return None
            return val

        parsed_params = {k: parse_val(v) for k, v in hyperparams.items()}
        self.knowledge = parsed_params.pop("knowledge", None)
  

        if self.knowledge is not None and os.path.exists(self.knowledge):
            self.knowledge = rlang.parse_file(self.knowledge.split("/")[-1])
            self.log("Knowledge loaded from:", self.rlang_file)

        # Instantiate and run training
        agent = AgentClass(self.env, knowledge=self.knowledge, **parsed_params)
        rewards = agent.train(episodes=15000)

        self.log(f"{algorithm} training complete. Episodes: {len(rewards)}")
        self.write_log_file()


# Entry point for subprocess from Tkinter
if __name__ == "__main__":
    # if len(sys.argv) < 6:
    #     print("Usage: python selector.py <rlang_file> <env_name> <algorithm> <hyperparams_json>")
    #     sys.exit(1)

    _, rlang_file, env_name, algorithm, hyperparams_json = sys.argv
 

    # Just the filename for knowledge
    rlang_file = os.path.basename(rlang_file)
    rlang_path = os.path.join(os.getcwd(), rlang_file)
    print()
    print("rlang")
    print(rlang_file)
    print("env name")
    print(env_name)
    print("algorithm")
    print(algorithm)
    print("hyper")
    print(hyperparams_json)
    print()
 
    
    try:
        hyperparams = json.loads(hyperparams_json)
    except json.JSONDecodeError:
        print("[ERROR] Could not parse hyperparams JSON.")
        sys.exit(1)



    # Initialize and run validator
    validator = RLangValidator(env_name, rlang_path)
    validator.run(algorithm,hyperparams)
