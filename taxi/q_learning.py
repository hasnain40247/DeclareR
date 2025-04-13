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
        super().__init__(env,env_name=env_name,knowledge=knowledge, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay)

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
                        self.q_table[s,a] = reward_function(state=VectorState([taxi_row, taxi_col, passenger_location, destination]), action=i)
            
            if transition_function and reward_function:
                for s in states:
                    taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(s)
                    for i, a in enumerate(actions):
                        s_primei = transition_function(state=VectorState([taxi_row, taxi_col, passenger_location, destination]), action=i)
                        if s_primei:
                            r_prime = self.weighted_reward(reward_function, s_primei, action=i)
                            v_s_prime = self.weighted_value(self.q_table, s_primei, actions)
                            self.q_table[s,a] += self.alpha * (r_prime + self.gamma * v_s_prime)
    
def plot_training_rewards_comparison(rewards_list, labels=None, colors=None, window_size=100, save_path="training_rewards.png"):
    """
    Plot multiple reward arrays on the same plot
    
    Args:
        rewards_list: List of reward arrays to plot
        labels: List of labels for each reward array
        colors: List of colors for each reward array
        window_size: Window size for moving average
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    
    if labels is None:
        labels = [f"Rewards {i+1}" for i in range(len(rewards_list))]
    
    if colors is None:
        colors = ['blue', 'green', 'purple', 'orange', 'brown']  # Default colors
    
    for i, rewards in enumerate(rewards_list):
        episodes = np.arange(len(rewards))
        
        # Plot raw rewards with low alpha
        plt.plot(episodes, rewards, label=f"{labels[i]} (Raw)", 
                 alpha=0.2, color=colors[i % len(colors)])
        
        # Calculate and plot smoothed rewards
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(episodes[:len(smoothed_rewards)], smoothed_rewards, 
                 label=f"{labels[i]} (MA, window={window_size})", 
                 color=colors[i % len(colors)])
    
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Comparison of Training Rewards")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    print(f"Plot saved as {save_path}")
    
    plt.close() 

if __name__ == '__main__':
    env = gym.make("Taxi-v3")
 
    # knowledge = rlang.parse_file("./taxi.rlang")
    # agent_with_policy = RLangQLearningAgent(env, knowledge=knowledge)
    # rewards_with_policy = agent_with_policy.train(episodes=800)
    # agent = RLangQLearningAgent(env,knowledge=None)
    # rewards = agent.train(episodes=800)
    # # print(f"Training complete. policy Average reward: {agent_with_policy.test(10)}")

    # # print(f"Training complete. Average reward: {agent.test(10)}")
    # agent.plot_training_rewards(rewards_with_policy,save_path="2q_learning_training_rewards_knowledge.png")
    # agent.plot_training_rewards(rewards,save_path="2q_learning_training_rewards.png")


    knowledge = rlang.parse_file("./taxi.rlang")
    agent_with_policy = RLangQLearningAgent(env, knowledge=knowledge)
    rewards_with_policy = agent_with_policy.train(episodes=15000)
    agent_with_policy_session =  agent_with_policy.test(10)
    average_reward = agent_with_policy_session[0]
    episode_descriptions = agent_with_policy_session[1]
    with open('episode_description_policy.pkl', 'wb') as f:
        pickle.dump(episode_descriptions, f)

    print(f"Training complete. Average reward: {average_reward}")

    agent = RLangQLearningAgent(env)
    rewards = agent.train(episodes=15000)
    agent_session =  agent.test(10)
    average_reward = agent_session[0]
    episode_descriptions = agent_session[1]
    with open('episode_description_wo_policy.pkl', 'wb') as f:
        pickle.dump(episode_descriptions, f)

    print(f"Training complete. Average reward: {average_reward}")
    
    # agent.plot_training_rewards(rewards_with_policy,save_path="./plots/q_learning_training_rewards_knowledge.png")
    # agent.plot_training_rewards(rewards,save_path="./plots/q_learning_training_rewards.png")

    plot_training_rewards_comparison(rewards_list = [rewards_with_policy, rewards], save_path = "q_learning_comparison.png")