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
import json

class BaseRLangQLearningAgent:
    def __init__(self, env, env_name="taxi", knowledge=None, alpha=0.9, gamma=0.9, epsilon=1, epsilon_decay=0.0001):
        self.env = env
        self.env_name = env_name

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table as a NumPy array
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        self.knowledge = knowledge
        self.training_details = []

    def weighted_reward(self, r_func, state_dict, action):
        return sum(r_func(state=VectorState(k), action=action) * v for k, v in state_dict.items())

    def weighted_value(self, q_func, state_dict, actions):
        # return sum(max(q_func[k, a] for a in actions) * v for k, v in state_dict.items())
        return sum(np.max([q_func[k, a] for a in actions]) * v for k, v in state_dict.items())


    def train(self, episodes, reward_callback=None):
        if self.knowledge:
            self.preload_knowledge()

        rewards_per_episode = np.zeros(episodes)
        rng = np.random.default_rng()

        for i in tqdm(range(episodes), desc="Training Progress", ncols=100):
            state = self.env.reset()[0]

            episode_details = {
                'episode': i,
           
                'q_table': self.q_table.tolist()  

            }

            terminated, truncated, rewards = False, False, 0

            while not (terminated or truncated):
                if rng.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                new_state, reward, terminated, truncated, _ = self.env.step(action)
                rewards += reward

              

                max_q = np.max(self.q_table[new_state])
                self.q_table[state, action] += self.alpha * (reward + self.gamma * max_q - self.q_table[state, action])
                state = new_state


            for key, value in episode_details.items():
                if isinstance(value, list):
                    episode_details[key] = [int(v) if isinstance(v, np.int64) else v for v in value]
                elif isinstance(value, np.int64):
                    episode_details[key] = int(value)
            self.epsilon = max(self.epsilon - self.epsilon_decay, 0)
            self.alpha = 0.0001 if self.epsilon == 0 else self.alpha
            rewards_per_episode[i] = rewards

            print(f"Episode {i}: Total Reward: {rewards}")

            if reward_callback:
                reward_callback(rewards)

        self.env.close()
        self.training_details.append(episode_details)


        with open(f"./training_details.json", "w") as f:
            json.dump(self.training_details, f)
        return rewards_per_episode

    def test(self, episodes=10, render=True):
        self.env = gym.make(self.env.spec.id, render_mode='human')
    
        pygame.display.set_mode((500, 500))

        rewards_per_episode = np.zeros(episodes)
        for i in range(episodes):
            state = self.env.reset()[0]
        
            terminated, truncated, rewards = False, False, 0
            
            while not (terminated or truncated):
                action = np.argmax(self.q_table[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                rewards += reward
                
            rewards_per_episode[i] = rewards
            
        if render:
            self.env.close()
            self.env = gym.make(self.env.spec.id)
            self.env.reset()
        
        self.env.close()
        pygame.quit()
        print(f"Average reward over {episodes} test episodes: {np.mean(rewards_per_episode)}")
        return np.mean(rewards_per_episode)

    def plot_training_rewards(self, rewards, window_size=100, save_path="training_rewards.png"):
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