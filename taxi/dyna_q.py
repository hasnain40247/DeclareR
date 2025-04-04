import numpy as np
import gymnasium as gym
import random
from collections import defaultdict
from tqdm import tqdm
import rlang
from rlang.grounding.utils.primitives import VectorState
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pygame
import os
import sys
three_folders_up = os.path.abspath(os.path.join(__file__, f"../../agents/"))
print(three_folders_up)
sys.path.append(three_folders_up)
from base_dyna_q import BaseDynaQAgent

class RLangDynaQAgent(BaseDynaQAgent):
    def __init__(self, env, n_planning_steps, alpha=0.1, gamma=0.99, epsilon=0.1, knowledge=None,policy_name=None, p_policy=0.2):
       super().__init__(env, n_planning_steps, alpha=0.1, gamma=0.99, epsilon=0.1, knowledge=knowledge,policy_name=policy_name, p_policy=0.2)
    def preload_knowledge(self):
        q_func = defaultdict(lambda: defaultdict(lambda: 0))
        reward_function = self.knowledge.reward_function
        transition_function = self.knowledge.transition_function
        states = range(self.env.observation_space.n)
        actions = range(self.env.action_space.n)
        
        if reward_function:
            for s in states:
                taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(s)
                state_vector = VectorState([taxi_row, taxi_col, passenger_location, destination])
                for a in actions:
                    q_func[s][a] = reward_function(state=state_vector, action=a)
        
        if transition_function and reward_function:
            for s in states:
                taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(s)
                state_vector = VectorState([taxi_row, taxi_col, passenger_location, destination])
                for a in actions:
                    s_primei = transition_function(state=state_vector, action=a)
                    if s_primei:
                        r_prime = self.weighted_reward(reward_function, s_primei, a)
                        v_s_prime = self.weighted_value(q_func, s_primei, actions)
                        q_func[s][a] += self.alpha * (r_prime + self.gamma * v_s_prime)
        
        return defaultdict(lambda: np.zeros(self.env.action_space.n), {
            state: np.array([q_func[state][a] for a in actions]) for state in states
        })

    def select_action(self, state):
            if self.knowledge and random.random() < self.p_policy:
                taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(state)
                state_vector = VectorState([taxi_row, taxi_col, passenger_location, destination])
                action = self.policy(state=state_vector)
                return int(list(action.keys())[0][0])
            
            if random.random() < self.epsilon:
                return self.env.action_space.sample()
            return np.argmax(self.q_table[state])
        


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    knowledge = rlang.parse_file("./taxi.rlang")
    agent_with_policy = RLangDynaQAgent(env,policy_name="taxi_policy", n_planning_steps=50, knowledge=knowledge, p_policy=0.7)
    rewards_with_policy = agent_with_policy.train(episodes=1500)
    print(f"Average reward with policy: {agent_with_policy.test(10)}")
    agent = RLangDynaQAgent(env, n_planning_steps=50)
    rewards = agent.train(episodes=1500)
    print(f"Average reward without policy: {agent.test(10)}")
    agent.plot_training_rewards(rewards_with_policy,save_path="./plots/dyna_q_training_rewards_knowledge.png")
    agent.plot_training_rewards(rewards,save_path="./plots/dyna_q_training_rewards.png")


