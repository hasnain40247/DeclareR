
import numpy as np
from rlang.grounding.utils.primitives  import VectorState
from tqdm import tqdm
import gym
import rlang
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pygame
import json
# Hyperparameters
# Cliff Walking
# self, env, knowledge=None, num_states=500, num_actions=6, r_max=20, gamma=0.95, delta=0.01, M=1
# Taxi
# self, env, knowledge=None, num_states=500, num_actions=6, r_max=20, gamma=0.95, delta=0.01, M=1):

class BaseRLangRMaxAgent:
     def __init__(self, env, knowledge=None, r_max=20, gamma=0.95, delta=0.01, M=1):

        self.knowledge = knowledge
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.actions = list(range(self.num_actions))

        self.r_max = r_max
        self.gamma = gamma
        self.delta = delta
        self.M = M
        self.env = env
        self.training_details = []  


        self.initialize_mdp()

     def initialize_mdp(self):
        self.emp_transition_dist = np.ones((self.num_actions, self.num_states, self.num_states)) / self.num_states
        self.emp_reward_dist = np.ones([self.num_states, self.num_actions]) * self.r_max

        self.state_action_counter_r = np.zeros([self.num_states, self.num_actions])
        self.emp_total_reward = np.zeros([self.num_states, self.num_actions])
        self.state_action_counter_t = np.zeros([self.num_states, self.num_actions])
        self.transition_count = np.zeros([self.num_actions, self.num_states, self.num_states])
        self.preload_knowledge()
    

     def compute_near_optimal_value_function(self):
        value_function = np.zeros([self.num_states, 1])
        convergence_delta = self.delta + 1

        while convergence_delta > self.delta:
            new_value_function = self.value_iteration_step(value_function)
            convergence_delta = np.max(np.abs(new_value_function - value_function))
            value_function = new_value_function

        action_value_function = np.zeros((self.num_states, self.num_actions))
        for i_action in range(self.num_actions):
            action_value_function[:, i_action] = (
                self.emp_reward_dist[:, i_action] +
                self.gamma * (self.emp_transition_dist[i_action, :, :] @ value_function).flatten()
            )
        return action_value_function

     def value_iteration_step(self, value_function):
        poss_values = np.zeros([self.num_states, self.num_actions])
        for i_action in range(self.num_actions):
            poss_values[:, i_action] = (
                self.emp_reward_dist[:, i_action] +
                self.gamma * (self.emp_transition_dist[i_action, :, :] @ value_function).flatten()
            )
        return np.expand_dims(np.max(poss_values, axis=1), axis=1)

     def select_action(self, q_optimal, state, deterministic=True):
        return np.argmax(q_optimal[state])

     def update_transition_model(self, state, action, new_state):
        if self.state_action_counter_t[state, action] < self.M:
            self.transition_count[action, state, new_state] += 1
            self.state_action_counter_t[state, action] += 1

            self.emp_transition_dist[action, state, :] = (
                self.transition_count[action, state, :] / self.state_action_counter_t[state, action]
            )

     def update_reward_model(self, state, action, reward):
        if self.state_action_counter_r[state, action] < self.M:
            self.emp_total_reward[state, action] += reward
            self.state_action_counter_r[state, action] += 1
            self.emp_reward_dist[state, action] = (
                self.emp_total_reward[state, action] / self.state_action_counter_r[state, action]
            )

     def update_empirical_mdp(self):
        for state in range(self.num_states):
            for action in range(self.num_actions):
                if self.state_action_counter_r[state, action] >= self.M and self.state_action_counter_t[state, action] >= self.M:
                    continue 

                self.emp_reward_dist[state, action] = self.r_max
                self.emp_transition_dist[action, state, :] = np.ones(self.num_states) / self.num_states


     def train(self, episodes=15000, max_steps=100):
        all_rewards = []

        for episode in tqdm(range(episodes)):
            state, info = self.env.reset()
            episode_details = {
                'episode': episode,
            
               
            }
            rewards = []

            for step in range(max_steps):
                q_optimal = self.compute_near_optimal_value_function()
                action = self.select_action(q_optimal, state)
                new_state, reward, terminated, truncated, _ = self.env.step(action)
        
                self.update_transition_model(state, action, new_state)
                self.update_reward_model(state, action, reward)

                rewards.append(reward)
                if terminated:
                    break
                state = new_state
            self.update_empirical_mdp()

            episode_reward = sum(rewards)
            episode_details["q_table"]=q_optimal.tolist() 
            for key, value in episode_details.items():
                if isinstance(value, list):
                    episode_details[key] = [int(v) if isinstance(v, np.int64) else v for v in value]
                elif isinstance(value, np.int64):
                    episode_details[key] = int(value) 
 

            print(f"Episode {episode}: Total Reward: {episode_reward}")  

            all_rewards.append(episode_reward)

        self.training_details.append(episode_details)

        with open(f"./training_details.json", "w") as f:
            json.dump(self.training_details, f)
        return all_rewards
    
     def test(self, num_episodes=10,render=True):
        self.env = gym.make(self.env.spec.id, render_mode='rgb_array')


        total_rewards = []  
      
        for episode in range(num_episodes):
            state = self.env.reset()[0]  
            total_reward = 0
            q_optimal = self.compute_near_optimal_value_function()
            terminated=False
            truncated=False
    
            while not terminated and not truncated:

                self.env.render()  
                action = self.select_action(q_optimal, state)  
                new_state, reward, terminated, truncated, _ = self.env.step(action)

                total_reward += reward
                state = new_state

            total_rewards.append(total_reward)

        if render:
            self.env.close()
            self.env = gym.make(self.env.spec.id)
            self.env.reset()
        
        self.env.close()
  

        return np.mean(total_rewards)
    