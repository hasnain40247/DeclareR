import numpy as np
from rlang.grounding.utils.primitives  import VectorState
from tqdm import tqdm
import gymnasium as gym
import rlang
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pygame
import os
import sys
from utils import plot_training_rewards, plot_comparison_training_rewards
three_folders_up = os.path.abspath(os.path.join(__file__, f"../../agents/"))
print(three_folders_up)
sys.path.append(three_folders_up)
from base_r_max import BaseRLangRMaxAgent

class RLangRmaxAgent(BaseRLangRMaxAgent):
    def __init__(self, env, knowledge=None, num_states=500, num_actions=6, r_max=20, gamma=0.95, delta=0.01, M=1):
        super().__init__(env, knowledge=knowledge, num_states=500, num_actions=6, r_max=20, gamma=0.95, delta=0.01, M=1)

    def state_to_vector(self, state):
        width = self.env.unwrapped.shape[1]
        return [state % width, state // width]
    
    def preload_knowledge(self):
        if not self.knowledge:
            return

        for state in range(self.num_states):
            for action in range(self.num_actions):
                try:
                    vector_state = VectorState(self.state_to_vector(state))
                    reward = int(self.knowledge.reward_function(state=vector_state, action=action)[0])
                    next_state_dist = self.knowledge.transition_function(state=vector_state, action=action)
                except AttributeError:
                    reward = self.r_max
                    next_state_dist = {}

                self.emp_reward_dist[state, action] = reward
                for next_state, prob in next_state_dist.items():
                    row, col = next_state
                    nrows, ncols = self.env.unwrapped.shape
                    if not (0 <= row < nrows and 0 <= col < ncols):
                        continue  # skip invalid symbolic prediction
                    encoded_state = row * ncols + col
                    self.emp_transition_dist[action, state, encoded_state] = prob



if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
    knowledge = rlang.parse_file("./frozen_lake.rlang")
    agent_with_policy = RLangRmaxAgent(env,knowledge=knowledge)
    rewards_with_policy = agent_with_policy.train(n_episodes=200)
    print(f"Average reward with policy: {agent_with_policy.test(10)}")
    # agent = RLangRmaxAgent(env)
    # rewards = agent.train(n_episodes=50)
    # print(f"Average reward without policy: {agent.test(10)}")
    # plot_training_rewards(rewards_with_policy,save_path="./plots/rmax_training_rewards_knowledge.png")
    # plot_training_rewards(rewards,save_path="./plots/rmax_training_rewards.png")
    # plot_comparison_training_rewards(
    #     reward_dict={
    #         "With RLang Policy": rewards_with_policy,
    #         "Without RLang": rewards
    #     },
    #     save_path="./plots/rmax_learning_comparison.png"
    # )




