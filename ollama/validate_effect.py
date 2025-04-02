# validate_effect.py
import sys
import gymnasium as gym
import rlang
from rlang.grounding.utils.primitives import VectorState
from rlang.agents.RLangPolicyAgentClass import RLangPolicyAgent
from collections import defaultdict
from datetime import datetime
import os
log_lines = []

def log(*args, print_to_console=True):
    message = "[LOG] " + " ".join(str(arg) for arg in args)
    log_lines.append(message)
    if print_to_console:
        print(message)

def write_log_file():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./logs/effect_validation_{timestamp}.log"
    with open(filename, "w") as f:
        f.write("\n".join(log_lines))
    print(f"[LOG] Saved log to {filename}")



def state_to_vector(self, state,env):
        if env=="taxi":
            taxi_row, taxi_col, passenger_location, destination = env.unwrapped.decode(state)
            return [taxi_row, taxi_col, passenger_location, destination]
        elif env=="cliff_walking":
            width = self.env.unwrapped.shape[1]
            return [state % width, state // width]
def validate(env,rlang_file):
    try:
        log(f"Parsing file: {rlang_file}")

        knowledge = rlang.parse_file(rlang_file)
        if env=="taxi":
            env = gym.make("Taxi-v3")
        elif env=="cliff_walking":
            env = gym.make("CliffWalking-v0")

        actions = range(env.action_space.n)
        states = range(env.observation_space.n)

        reward_func = knowledge.reward_function
        trans_func = knowledge.transition_function
        log(f"Reward Function → {reward_func}")
        log(f"Transition Function → {trans_func}")



        log("Beginning state-action logging...\n")
        for s in states:
            state_vec=state_to_vector(s)
            vstate = VectorState(state_vec)

            for a in actions:
                action_log = f"State={state_vec}, Action={a}"

                if reward_func:
                    try:
                        r = reward_func(state=vstate, action=a)
                        log(f"{action_log} → Reward = {r}")
                    except Exception as e:
                        log(f"{action_log} → Reward ERROR: {e}")

                if trans_func and reward_func:
                    try:
                        s_prime = trans_func(state=vstate, action=a)
                        log(f"{action_log} → Transition = {s_prime}")
                    except Exception as e:
                        log(f"{action_log} → Transition ERROR: {e}")

        print("VALID")

    except Exception as e:
        print(f"ERROR: {str(e)}")
    finally:
        write_log_file()

if __name__ == "__main__":
    validate(sys.argv[1])
