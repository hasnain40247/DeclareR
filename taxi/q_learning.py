import gymnasium as gym
import numpy as np
import pickle
from collections import defaultdict
from rlang.grounding.utils.primitives import VectorState
import rlang
from rlang.agents.RLangPolicyAgentClass import RLangPolicyAgent
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pygame
import os  
import json
import sys
three_folders_up = os.path.abspath(os.path.join(__file__, f"../../agents/"))
sys.path.append(three_folders_up)
from base_q_learning import BaseRLangQLearningAgent


class RLangQLearningAgent(BaseRLangQLearningAgent):
    def __init__(self, env,env_name="taxi",knowledge=None, alpha=0.9, gamma=0.9, epsilon=1, epsilon_decay=0.0001):
        super().__init__(env,env_name="taxi",knowledge=knowledge, alpha=0.9, gamma=0.9, epsilon=1, epsilon_decay=0.0001)

    def preload_knowledge(self):
        states = range(self.env.observation_space.n)
        actions = range(self.env.action_space.n)
        if self.knowledge:
            reward_function = self.knowledge.reward_function
            transition_function = self.knowledge.transition_function
            
            if reward_function:
                for s in states:
                    taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(s)
                    for i, a in enumerate(actions):
                        self.q_table[s][a] = reward_function(state=VectorState([taxi_row, taxi_col, passenger_location, destination]), action=i)
            
            if transition_function and reward_function:
                for s in states:
                    taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(s)
                    for i, a in enumerate(actions):
                        s_primei = transition_function(state=VectorState([taxi_row, taxi_col, passenger_location, destination]), action=i)
                        if s_primei:
                            r_prime = self.weighted_reward(reward_function, s_primei, action=i)
                            v_s_prime = self.weighted_value(self.q_table, s_primei, actions)
                            self.q_table[s,a] += self.alpha * (r_prime + self.gamma * v_s_prime)
    
    

if __name__ == '__main__':
    env = gym.make("Taxi-v3")
    agent = RLangQLearningAgent(env,env_name="taxi")
    rewards = agent.train(episodes=10000)
    print(f"Training complete. Average reward: {agent.test(10)}")
    #     np.set_printoptions(threshold=np.inf) 

    # knowledge = rlang.parse_file("./taxi.rlang")
    # agent_with_policy = RLangQLearningAgent(env, knowledge=knowledge)
    # rewards_with_policy = agent_with_policy.train(episodes=15000)
    # print(f"Training complete. Average reward: {agent_with_policy.test(10)}")
    # agent = RLangQLearningAgent(env)
    # rewards = agent.train(episodes=15000)
    # print(f"Training complete. Average reward: {agent.test(10)}")
    # agent.plot_training_rewards(rewards_with_policy,save_path="./plots/q_learning_training_rewards_knowledge.png")
    # agent.plot_training_rewards(rewards,save_path="./plots/q_learning_training_rewards.png")


