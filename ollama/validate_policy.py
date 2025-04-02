# validate_effect.py
import sys
import gymnasium as gym
import rlang
from rlang.grounding.utils.primitives import VectorState
from rlang.agents.RLangPolicyAgentClass import RLangPolicyAgent
from collections import defaultdict
from datetime import datetime

log_lines = []

def log(*args, print_to_console=True):
    message = "[LOG] " + " ".join(str(arg) for arg in args)
    log_lines.append(message)
    if print_to_console:
        print(message)

def write_log_file():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./logs/policy_validation_{timestamp}.log"
    with open(filename, "w") as f:
        f.write("\n".join(log_lines))
    print(f"[LOG] Saved log to {filename}")

def validate(rlang_file,policy_name):
    try:
        log(f"Parsing file: {rlang_file}")

        knowledge = rlang.parse_file(rlang_file.split("/")[-1])
        taxi_policy = knowledge[f'{policy_name}']

        env = gym.make("Taxi-v3")
 
        states = range(env.observation_space.n)

        log(f"Policy Function → {taxi_policy}")
  

        log("Beginning state-action logging...\n")
        for s in states:
            taxi_row, taxi_col, passenger_location, destination = env.unwrapped.decode(s)
            state_vec = [taxi_row, taxi_col, passenger_location, destination]
            vstate = VectorState(state_vec)
    
            if taxi_policy:
                try:
                    action = taxi_policy(state=vstate)
                    action_log = f"State={state_vec}, Action={action}"
                    log(f"{action_log}")
                except Exception as e:
                    log(f"{action_log} → ERROR: {e}")

        print("VALID")

    except Exception as e:
        print(f"ERROR: {str(e)}")
    finally:
        write_log_file()

if __name__ == "__main__":
    file_path = sys.argv[1]
    policy_name = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    validate(file_path, policy_name)
  
