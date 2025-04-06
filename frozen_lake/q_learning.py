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
import os  # To force quit Pygame if needed
# from utils import plot_training_rewards, plot_comparison_training_rewards
import json
import sys
three_folders_up = os.path.abspath(os.path.join(__file__, f"../../agents/"))
sys.path.append(three_folders_up)
from base_q_learning import BaseRLangQLearningAgent

class RLangQLearningAgent(BaseRLangQLearningAgent):
    def __init__(self, env,env_name="cliff_walking", knowledge=None, alpha=0.9, gamma=0.9, epsilon=1, epsilon_decay=0.0001):
        super().__init__(env,env_name=env_name,knowledge=knowledge, alpha=0.9, gamma=0.9, epsilon=1, epsilon_decay=0.0001)
      
    def state_to_vector(self, state):
        width = self.env.unwrapped.shape[1]
        return np.array([state % width, state // width])

    def preload_knowledge(self):
        states = range(self.env.observation_space.n)
        actions = range(self.env.action_space.n)
        if self.knowledge:
            reward_function = self.knowledge.reward_function
            transition_function = self.knowledge.transition_function
            if reward_function:
                for s in states:
                    vector_s = self.state_to_vector(s)
                    for i, a in enumerate(actions):
                        self.q_table[s, a] = reward_function(state=VectorState(vector_s), action=i)

            if transition_function and reward_function:
                for s in states:
                    vector_s = self.state_to_vector(s)
                    for a in actions:
                        s_prime_dist = transition_function(state=VectorState(vector_s), action=a)
                        if s_prime_dist:
                            r_prime = self.weighted_reward(reward_function, s_prime_dist, action=a)
                            v_s_prime = self.weighted_value(self.q_table, s_prime_dist, actions)
                            self.q_table[s, a] += self.alpha * (r_prime + self.gamma * v_s_prime)


    
    
if __name__ == '__main__':
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
    # agent = RLangQLearningAgent(env,env_name="cliff_walking")
    # rewards = agent.train(episodes=10000)
    # np.set_printoptions(threshold=np.inf) 

    knowledge = rlang.parse_file("./cliff_walking.rlang")
    print("knowledge")
    print(knowledge)
    agent_with_policy = RLangQLearningAgent(env,env_name="cliff_walking", knowledge=knowledge,epsilon=1,epsilon_decay=0.02)
    rewards_with_policy = agent_with_policy.train(episodes=100)
    # print(f"Training complete. Average reward: {agent_with_policy.test(10)}")
    # agent = RLangQLearningAgent(env)
    # rewards = agent.train(episodes=100)
    # print(f"Training complete. Average reward: {agent.test(10)}")
    # plot_training_rewards(rewards_with_policy,save_path="./plots/q_learning_training_rewards_knowledge.png")
    # plot_training_rewards(rewards,save_path="./plots/q_learning_training_rewards.png")
    # plot_comparison_training_rewards(
    #     reward_dict={
    #         "With RLang Policy": rewards_with_policy,
    #         "Without RLang": rewards
    #     },
    #     save_path="./plots/q_learning_comparison.png"
    # )
