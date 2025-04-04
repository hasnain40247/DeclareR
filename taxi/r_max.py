
import numpy as np
from rlang.grounding.utils.primitives  import VectorState
from tqdm import tqdm
import gym
import rlang
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pygame
import os
import sys
three_folders_up = os.path.abspath(os.path.join(__file__, f"../../agents/"))
print(three_folders_up)
sys.path.append(three_folders_up)

from base_r_max import BaseRLangRMaxAgent

class RLangRMaxAgent(BaseRLangRMaxAgent):
    def __init__(self, env, knowledge=None,  r_max=20, gamma=0.95, delta=0.01, M=1):
        super().__init__(env, knowledge=knowledge, r_max=20, gamma=0.95, delta=0.01, M=1)

    def preload_knowledge(self):
        if not self.knowledge:
            return

        for state in range(self.num_states):
            for action in range(self.num_actions):
                try:
                    decoded = self.env.unwrapped.decode(state)
                    vector_state = VectorState(list(decoded))

                    reward = int(self.knowledge.reward_function(state=vector_state, action=action)[0])

                    next_state_dist = self.knowledge.transition_function(state=vector_state, action=action)
                except AttributeError:
                    reward = self.r_max
                    next_state_dist = {}

                self.emp_reward_dist[state, action] = reward

                for next_state, prob in next_state_dist.items():
                    encoded_state = int(self.env.encode(*next_state))
                    self.emp_transition_dist[action, state, encoded_state] = prob



    def plot_training_rewards(self,rewards, window_size=100,save_path="training_rewards.png"):
        episodes = np.arange(len(rewards))
    
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards, label="Rewards per Episode", alpha=0.3)
        plt.plot(episodes[:len(smoothed_rewards)], smoothed_rewards, label=f"Moving Average (window={window_size})", color='red')
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Training Rewards Over Episodes")
        plt.legend()
        plt.grid()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        print(f"Plot saved as {save_path}")

        plt.close()  


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    knowledge = rlang.parse_file("./taxi.rlang")
    agent_with_policy = RLangRMaxAgent(env,knowledge=knowledge)
    rewards_with_policy = agent_with_policy.train(episodes=200)
    print(f"Average reward with policy: {agent_with_policy.test(10)}")
    agent = RLangRMaxAgent(env)
    rewards = agent.train(episodes=200)
    print(f"Average reward without policy: {agent.test(10)}")
    agent.plot_training_rewards(rewards_with_policy,save_path="./plots/rmax_training_rewards_knowledge.png")
    agent.plot_training_rewards(rewards,save_path="./plots/rmax_training_rewards.png")




