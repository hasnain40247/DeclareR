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
from utils import plot_training_rewards, plot_comparison_training_rewards

class DynaQAgent:
    def __init__(self, env, n_planning_steps, alpha=0.1, gamma=0.99, epsilon=0, knowledge=None, p_policy=0.2):
        self.env = env
        self.n_planning_steps = n_planning_steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.knowledge = knowledge
        self.p_policy = p_policy
        self.model = {}
        
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
        if knowledge:
            self.cliff_policy = knowledge['cliff_policy']
            self.q_table = self.initialize_q_table_with_knowledge()
    
    def state_to_vector(self, state):
        width = self.env.unwrapped.shape[1]
        return [state % width, state // width]
    
    def initialize_q_table_with_knowledge(self):
        q_func = defaultdict(lambda: defaultdict(lambda: 0))
        reward_function = self.knowledge.reward_function
        transition_function = self.knowledge.transition_function
        states = range(self.env.observation_space.n)
        actions = range(self.env.action_space.n)
        
        if reward_function:
            for s in states:
                vector_s = self.state_to_vector(s)
                for i, a in enumerate(actions):
                    q_func[s][a] = reward_function(state=VectorState(vector_s), action=i)

        if transition_function and reward_function:
            for s in states:
                vector_s = self.state_to_vector(s)
                for a in actions:
                    s_prime_dist = transition_function(state=VectorState(vector_s), action=a)
                    if s_prime_dist:
                        r_prime = self.weighted_reward(reward_function, s_prime_dist, action=a)
                        v_s_prime = self.weighted_value(q_func, s_prime_dist, actions)
                        q_func[s][a] += self.alpha * (r_prime + self.gamma * v_s_prime)
                        
        return defaultdict(lambda: np.zeros(self.env.action_space.n), {
            state: np.array([q_func[state][a] for a in actions]) for state in states
        })

    def weighted_reward(self, r_func, state_dict, action):
        return sum(r_func(state=VectorState(k), action=action) * v for k, v in state_dict.items())

    def weighted_value(self, q_func, state_dict, actions):
        return sum(max(q_func[k][a] for a in actions) * v for k, v in state_dict.items())
    
    def train(self, n_episodes=500):
        rewards = []
        for episode in tqdm(range(n_episodes)):
            state, info = self.env.reset()
            total_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.update_q_table(state, action, next_state, reward, done)
                self.update_model(state, action, next_state, reward, done)
                self.plan()
                state = next_state
                total_reward += reward
            
            rewards.append(total_reward)
        return rewards
    
    def select_action(self, state):
        if self.knowledge and random.random() < self.p_policy:
            vector_s = self.state_to_vector(state)
            action = self.cliff_policy(state=VectorState(vector_s))
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
    
    def test(self, num_episodes=10,render=True):
        self.env = gym.make('CliffWalking-v0', render_mode='rgb_array')
        # pygame.display.set_mode((500, 500))
        
        total_rewards = []
        for _ in range(num_episodes):
            state, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            while not (done or truncated):
                # self.env.render()  
                action = np.argmax(self.q_table[state])
                state, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
        
        if render:
            self.env.close()
            self.env = gym.make("CliffWalking-v0")
            self.env.reset()
        
        self.env.close()
        # pygame.quit()

        return np.mean(total_rewards)


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    np.set_printoptions(threshold=np.inf) 
    knowledge = rlang.parse_file("./cliff_walking.rlang")
    
    agent_with_policy = DynaQAgent(env, n_planning_steps=50, knowledge=knowledge, p_policy=0.1)
    rewards_with_policy = agent_with_policy.train(n_episodes=50)
    print(f"Average reward with policy: {agent_with_policy.test(10)}")
    agent = DynaQAgent(env, n_planning_steps=50)
    rewards = agent.train(n_episodes=50)
    print(f"Average reward without policy: {agent.test(10)}")
    plot_training_rewards(rewards_with_policy,save_path="./plots/dyna_q_training_rewards_knowledge.png")
    plot_training_rewards(rewards,save_path="./plots/dyna_q_training_rewards.png")
    plot_comparison_training_rewards(
        reward_dict={
            "With RLang Policy": rewards_with_policy,
            "Without RLang": rewards
        },
        save_path="./plots/dyna_q_learning_comparison.png"
    )

