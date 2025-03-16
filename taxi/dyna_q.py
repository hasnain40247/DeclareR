import numpy as np
import gymnasium as gym
import random
from collections import defaultdict
from tqdm import tqdm
import rlang
from rlang.grounding.utils.primitives import VectorState

class DynaQAgent:
    def __init__(self, env, n_planning_steps, alpha=0.1, gamma=0.99, epsilon=0.1, knowledge=None, p_policy=0.2):
        self.env = env
        self.n_planning_steps = n_planning_steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.knowledge = knowledge
        self.p_policy = p_policy
        self.model = {}
        
        if knowledge:
            self.taxi_policy = knowledge['taxi_policy']
            self.q_table = self.initialize_q_table_with_knowledge()
        else:
            self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    def initialize_q_table_with_knowledge(self):
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
    
    def test(self, num_episodes=100):
        self.env = gym.make(self.env.spec.id, render_mode="human")  
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
        
        self.env.close()
        return np.mean(total_rewards)
    
   

if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    knowledge = rlang.parse_file("./taxi.rlang")
    agent_with_policy = DynaQAgent(env, n_planning_steps=50, knowledge=knowledge, p_policy=0.7)
    rewards_with_policy = agent_with_policy.train(n_episodes=1500)
    print(f"Average reward with policy: {agent_with_policy.test(10)}")
    agent = DynaQAgent(env, n_planning_steps=50)
    rewards = agent.train(n_episodes=1500)
    print(f"Average reward without policy: {agent.test(10)}")
