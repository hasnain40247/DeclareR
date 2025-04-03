import sys
import os
import gymnasium as gym
from datetime import datetime
from collections import defaultdict
from rlang.grounding.utils.primitives import VectorState
import rlang





class RLangValidator:
    def __init__(self, env_name, rlang_file, policy_name=None):
        self.env_name = env_name.lower()
        self.rlang_file = rlang_file
        if policy_name=="0":
            self.policy_name = None
        else:
            self.policy_name = policy_name

        self.env = None
        self.knowledge = None
        self.log_lines = []

    def log(self, *args, print_to_console=True):
        message = "[LOG] " + " ".join(str(arg) for arg in args)
        self.log_lines.append(message)
        if print_to_console:
            print(message)

    def write_log_file(self, prefix="validation"):
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./logs/{prefix}.log"
        with open(filename, "w") as f:
            f.write("\n".join(self.log_lines))
        print(f"[LOG] Saved log to {filename}")

 

if __name__=="__main__":

    validation_script_path, rlang_file,env_name,algorithm=sys.argv
    rlang_file=rlang_file.split("/")[-1]

    if env_name=="taxi":
        env = gym.make("Taxi-v3")
    else:
        env = gym.make("CliffWalking-v0")

    three_folders_up = os.path.abspath(os.path.join(__file__, f"../../{env_name}"))




    sys.path.append(three_folders_up)

    from q_learning import RLangQLearningAgent




    if algorithm=="Q-Learning":
        knowledge = rlang.parse_file(rlang_file)
        # print("knowledge")
        # print(knowledge)

        # agent_with_policy = RLangQLearningAgent(env,env_name="cliff_walking", knowledge=knowledge,epsilon=1,epsilon_decay=0.02)
        # rewards_with_policy = agent_with_policy.train(episodes=100)
        # knowledge = rlang.parse_file("./taxi.rlang")
        agent_with_policy = RLangQLearningAgent(env, knowledge=knowledge)
        rewards_with_policy = agent_with_policy.train(episodes=15000)
        
    
    



    

    