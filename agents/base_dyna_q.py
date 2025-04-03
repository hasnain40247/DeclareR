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
import json
class BaseDynaQAgent:
    def __init__(self, env, n_planning_steps, alpha=0.1, gamma=0.99, epsilon=0.1, knowledge=None,policy_name=None, p_policy=0.2):
        self.env = env
        self.n_planning_steps = n_planning_steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.knowledge = knowledge
        self.p_policy = p_policy
        self.model = {}
        self.policy_name=policy_name
        self.training_details = []

        
       
       
        # self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def weighted_reward(self, r_func, state_dict, action):
        return sum(r_func(state=VectorState(k), action=action) * v for k, v in state_dict.items())

    def weighted_value(self, q_func, state_dict, actions):
        return sum(max(q_func[k][a] for a in actions) * v for k, v in state_dict.items())
    
    def train(self, n_episodes=500):

        if self.knowledge:
            self.taxi_policy = self.knowledge[self.policy_name]
            self.preload_knowledge()

        rewards = []
        for episode in tqdm(range(n_episodes)):
            state, info = self.env.reset()
            total_reward = 0
            done = False
            truncated = False
            episode_details = {
                'episode': episode,
                'states': [],
                'actions': [],
                'q_table': self.q_table.tolist()  

            }
            
            while not (done or truncated):
                action = self.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.update_q_table(state, action, next_state, reward, done)
                self.update_model(state, action, next_state, reward, done)
                self.plan()
                episode_details['states'].append(state)
                episode_details['actions'].append(action)
                state = next_state
                total_reward += reward
            for key, value in episode_details.items():
                if isinstance(value, list):
                    episode_details[key] = [int(v) if isinstance(v, np.int64) else v for v in value]
                elif isinstance(value, np.int64):
                    episode_details[key] = int(value)
            rewards.append(total_reward)
            self.training_details.append(episode_details)
            print(f"Episode {episode}: Total Reward: {total_reward}")


        with open(f"./training_details.json", "w") as f:
            json.dump(self.training_details, f)
        return rewards
    
    def select_action(self, state):
        if self.knowledge and random.random() < self.p_policy:
            taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(state)
            state_vector = VectorState([taxi_row, taxi_col, passenger_location, destination])
            action = self.taxi_policy(state=state_vector)
            return int(list(action.keys())[0][0])
        
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, next_state, reward, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def update_model(self, state, action, next_state, reward, done):
        self.model[(state, action)] = (next_state, reward, done)
    
    def plan(self):
        for _ in range(self.n_planning_steps):
            if self.model:
                s, a = random.choice(list(self.model.keys()))
                s_next, r, d = self.model[(s, a)]
                self.update_q_table(s, a, s_next, r, d)
    
    def test(self, num_episodes=100,render=True):
        self.env = gym.make(self.env.spec.id, render_mode="human")  
    
        pygame.display.set_mode((500, 500))

 
        total_rewards = []
        for _ in range(num_episodes):
            state, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            while not (done or truncated):
                self.env.render()  
                action = np.argmax(self.q_table[state])
                state, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
        
        if render:
            self.env.close()
            self.env = gym.make(self.env.spec.id)
            self.env.reset()
        
        self.env.close()
        pygame.quit()

        return np.mean(total_rewards)

   
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

