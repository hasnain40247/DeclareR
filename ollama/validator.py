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

    def load_env_and_knowledge(self):
        self.log(f"Parsing file: {self.rlang_file}")
        self.knowledge = rlang.parse_file(self.rlang_file)
        if self.env_name == "taxi":
            self.env = gym.make("Taxi-v3")
        elif self.env_name == "cliff_walking":
            self.env = gym.make("CliffWalking-v0")
        else:
            raise ValueError(f"Unsupported environment: {self.env_name}")

    def state_to_vector(self, s):
        if self.env_name == "taxi":
            taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(s)
            return [taxi_row, taxi_col, passenger_location, destination]
        elif self.env_name == "cliff_walking":
            width = self.env.unwrapped.shape[1]
            return [s % width, s // width]
        else:
            raise ValueError("Invalid environment")

    def validate_effect(self):
        try:
            self.load_env_and_knowledge()
            actions = range(self.env.action_space.n)
            states = range(self.env.observation_space.n)

            reward_func = self.knowledge.reward_function
            trans_func = self.knowledge.transition_function

            self.log(f"Reward Function → {reward_func}")
            self.log(f"Transition Function → {trans_func}")
            self.log("Beginning state-action logging...\n")

            for s in states:
                state_vec = self.state_to_vector(s)
                vstate = VectorState(state_vec)

                for a in actions:
                    context = f"State={state_vec}, Action={a}"
                    if reward_func:
                        try:
                            r = reward_func(state=vstate, action=a)
                            self.log(f"{context} → Reward = {r}")
                        except Exception as e:
                            self.log(f"{context} → Reward ERROR: {e}")

                    if trans_func:
                        try:
                            s_prime = trans_func(state=vstate, action=a)
                            self.log(f"{context} → Transition = {s_prime}")
                        except Exception as e:
                            self.log(f"{context} → Transition ERROR: {e}")

            print("VALID")

        except Exception as e:
            print(f"ERROR: {str(e)}")
        finally:
            self.write_log_file("effect_validation")

    def validate_policy(self):
        try:
            self.load_env_and_knowledge()

            if not self.policy_name:
                raise ValueError("Policy name is required for policy validation")

            policy = self.knowledge[self.policy_name]
            states = range(self.env.observation_space.n)

            self.log(f"Policy Function → {policy}")
            self.log("Beginning policy execution logging...\n")

            for s in states:
                state_vec = self.state_to_vector(s)
                vstate = VectorState(state_vec)
                try:
                    action = policy(state=vstate)
                    self.log(f"State={state_vec}, Action={action}")
                except Exception as e:
                    self.log(f"State={state_vec} → ERROR: {e}")

            print("VALID")

        except Exception as e:
            print(f"ERROR: {str(e)}")
        finally:
            self.write_log_file("policy_validation")


if __name__=="__main__":

    validation_script_path, rlang_file,env_name,is_effect,policy=sys.argv
    rlang_file=rlang_file.split("/")[-1]
    validator=RLangValidator(env_name,rlang_file,policy_name=policy)
    
 
    if int(is_effect):
        validator.validate_effect()

    else:
        validator.validate_policy()


    

    