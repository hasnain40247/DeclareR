import matplotlib
matplotlib.use = lambda *args, **kwargs: None

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import gym
import rlang
import graphviz
from sklearn.tree import export_graphviz
import scipy.special
from q_learning import RLangQLearningAgent

class GetExplainabilityPlotsForEnv:
    
    def __init__(self, env, Q_table, policy, states, actions):
        self.env = env
        self.Q_table = Q_table
        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n
        self.action_labels = ["South", "North", "East", "West", "Pickup", "Dropoff"]
        
        self.policy = policy
        self.states = states
        self.actions = actions
        self.grid_size = 5
        
        X_train = np.array(states)
        y_train = np.array(actions)
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.action_symbols = {
            0: "S",  # South
            1: "N",  # North
            2: "E",  # East
            3: "W",  # West
            4: "P",  # Pickup
            5: "D"   # Dropoff
        }
                
        # Initialize and train models
        self.dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=4, random_state=21)
        self.log_reg_model = LogisticRegression(max_iter=1000)

        self.dt_model.fit(X_train, y_train)
        self.log_reg_model.fit(X_train, y_train)
        
        # Evaluate how well the models are imitating the Q-learning agent
        self.evaluate_models()

    def evaluate_models(self):
        correct_dt_predictions = 0
        correct_lr_predictions = 0

        for state in range(self.state_dim):
            true_action = np.argmax(self.policy[state])  # True action taken by Q-learning agent
            taxi_row, taxi_col, passenger_loc, destination = self.env.decode(state)  # Get the state representation
            state_rep = (taxi_row, taxi_col, passenger_loc, destination)

            # Predict action using Decision Tree and Logistic Regression
            predicted_action_dt = self.dt_model.predict([state_rep])
            predicted_action_lr = self.log_reg_model.predict([state_rep])

            # Map numeric prediction to action names
            predicted_action_name_dt = self.action_labels[predicted_action_dt[0]]
            predicted_action_name_lr = self.action_labels[predicted_action_lr[0]]

            # Check if the predicted action matches the true action
            if predicted_action_dt == true_action:
                correct_dt_predictions += 1
            if predicted_action_lr == true_action:
                correct_lr_predictions += 1

        # Calculate the imitation accuracy
        dt_accuracy = correct_dt_predictions / self.state_dim
        lr_accuracy = correct_lr_predictions / self.state_dim

        print(f"Decision Tree imitation accuracy: {dt_accuracy * 100}%")
        print(f"Logistic Regression imitation accuracy: {lr_accuracy * 100}%")

    def run_episode(self, Q_table, env):
        """Runs one episode using the learned Q-table and logs the trajectory."""
        state = env.reset()[0]
        done = False

        trajectory = {
            "states": [],
            "actions": [],
            "rewards": []
        }

        while not done:
            action = np.argmax(Q_table[state, :])  # Always take the best action
            next_state, reward, done, _, _ = env.step(action)

            # Log data
            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)

            state = next_state

        return trajectory
    
    def plot_action_trajectory(self, trajectory):
        """Plots the action trajectory over time."""
        actions = trajectory["actions"]

        plt.figure(figsize=(10, 3))
        plt.step(range(len(actions)), actions, where='mid', linestyle='-', marker='o', color='r')
        plt.yticks(range(self.action_dim), self.action_labels)
        plt.xlabel("Time Step")
        plt.ylabel("Action")
        plt.title("Action Trajectory Over Time")
        plt.grid()
        plt.savefig("plots/episode_action_trajectory.png")
        plt.close()

    def plot_trajectory_state_visits(self, trajectory):
        """Plots the state visit heatmap."""
        visits = np.zeros((self.grid_size, self.grid_size))
        for state in trajectory["states"]:
            taxi_row, taxi_col, _, _ = self.env.decode(state)
            visits[taxi_row, taxi_col] += 1 

        plt.figure(figsize=(6, 6))
        sns.heatmap(visits, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("State Visit Heatmap")
        plt.xlabel("Taxi Column")
        plt.ylabel("Taxi Row")
        plt.savefig("plots/episode_statevisits_trajectory.png")
        plt.close()

    def plot_trajectory_rewards(self, trajectory):
        """Plots the reward trajectory over time."""
        rewards = trajectory["rewards"]
        plt.figure(figsize=(10, 3))
        plt.step(range(len(rewards)), rewards, where='mid', linestyle='-', marker='o', color='g')
        plt.xlabel("Time Step")
        plt.ylabel("Reward")
        plt.title("Reward Trajectory Over Time")
        plt.grid()
        plt.savefig("plots/episode_rewards_trajectory.png")
        plt.close()

    def visualize_policy(self):
        """Visualizes the learned policy for the Taxi-v3 environment."""
        
        grid_size = (self.grid_size, self.grid_size)
        policy_grid = np.full(grid_size, " ", dtype="<U2")  # Initialize empty grid

        # Extract optimal actions for each taxi position
        for state in range(self.state_dim):  # 500 states in Taxi-v3
            taxi_row, taxi_col, pass_loc, dest_loc = self.env.decode(state)
            best_action = np.argmax(self.Q_table[state, :])  # Get best action from Q-table
            policy_grid[taxi_row, taxi_col] = self.action_symbols[best_action]

        plt.figure(figsize=(6, 6))
        plt.imshow(np.zeros(grid_size), cmap="Blues", vmin=-1, vmax=1)
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                plt.text(col, row, policy_grid[row, col], ha='center', va='center', fontsize=14, color='black')

        plt.title("Policy Visualization")
        plt.axis('off')
        plt.savefig("plots/taxi_policy.png")
        plt.close()

    def plot_feature_importance(self):
        """Plots feature importance from a Decision Tree model."""
        feature_importance = self.dt_model.feature_importances_
        features = ["taxi_row", "taxi_col", "passenger_loc", "destination"]

        plt.figure(figsize=(8, 5))
        plt.bar(features, feature_importance, color='skyblue')
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance from Decision Tree")
        plt.savefig("plots/feature_importance_decision_tree.png")
        plt.close()

    def shap_summary_plot(self):
        """Generates SHAP summary plot for Logistic Regression model."""
        explainer = shap.KernelExplainer(self.log_reg_model.predict_proba, self.X_train)  # Use the predict_proba method
        shap_values = explainer.shap_values(self.X_train)

        shap.summary_plot(shap_values, self.X_train, feature_names=["taxi_row", "taxi_col", "passenger_loc", "destination"], class_names=self.action_labels)

    
    def get_state_representation(self, state_index):
        taxi_row, taxi_col, passenger_loc, destination = self.env.decode(state_index)
        return (taxi_row, taxi_col, passenger_loc, destination)

    def plot_probability_heatmap(self):
        """Plots a heatmap of predicted probabilities for each state-action pair."""
        probabilities = []

        for state in range(self.state_dim):
            q_values = self.Q_table[state]
            exp_q_values = np.exp(q_values - np.max(q_values))  # Numerical stability
            probabilities_state = exp_q_values / np.sum(exp_q_values)
            probabilities.append(probabilities_state)

        probabilities = np.array(probabilities)
        probabilities_df = pd.DataFrame(probabilities, columns=self.action_labels)

        state_representations = [self.get_state_representation(state) for state in range(500)]
        state_labels = [str(state_rep) for state_rep in state_representations]

        plt.figure(figsize=(20, 80))
        sns.heatmap(probabilities_df, 
                    cmap='coolwarm', 
                    annot=True, fmt=".2f", 
                    cbar=True, 
                    xticklabels=self.action_labels, 
                    yticklabels=state_labels)

        plt.title("Predicted Probabilities of Actions for Each State")
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.tight_layout()
        plt.savefig("plots/Q_table_heatmap.png")
        plt.close()
    
    def plot_logisticregression_probability_heatmap(self):
        """Plots a heatmap of predicted probabilities for each state-action pair using logistic regression."""
        probabilities = []

        for state in range(self.state_dim):
            taxi_row, taxi_col, passenger_loc, destination = self.env.decode(state)
            state_features = np.array([taxi_row, taxi_col, passenger_loc, destination]).reshape(1, -1)

            predicted_probs = self.log_reg_model.predict_proba(state_features)[0]
            probabilities.append(predicted_probs)

        probabilities = np.array(probabilities)
        probabilities_df = pd.DataFrame(probabilities, columns=self.action_labels)

        state_representations = [self.get_state_representation(state) for state in range(500)]
        state_labels = [str(state_rep) for state_rep in state_representations]

        plt.figure(figsize=(20, 80))
        sns.heatmap(probabilities_df, 
                    cmap='coolwarm', 
                    annot=True, fmt=".2f", 
                    cbar=True, 
                    xticklabels=self.action_labels, 
                    yticklabels=state_labels)

        plt.title("Predicted Probabilities of Actions for Each State (Logistic Regression)")
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.tight_layout()
        plt.savefig("plots/logistic_regression_heatmap.png")
        plt.close()

    def visualize_decision_tree(self):
        """Visualizes decision tree classifier.""" 
        dot_data = export_graphviz(self.dt_model, 
                                   out_file=None,  
                                   feature_names=["taxi_row", "taxi_col", "passenger_loc", "destination"],  
                                   class_names=self.action_labels,  
                                   filled=True, rounded=True,  
                                   special_characters=True)  
        graph = graphviz.Source(dot_data)  
        graph.render("plots/decision_tree")  # Generates decision tree as a pdf file
        graph.view()
        
        
class LimeExplainer:
    def __init__(self, env, Q_table):
        self.env = env
        self.Q_table = Q_table
        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n
        self.action_labels = ["South", "North", "East", "West", "Pickup", "Dropoff"]
        self.action_symbols = {
            0: "South",  
            1: "North",  
            2: "East",  
            3: "West",  
            4: "Pickup",  
            5: "Dropoff"
        }

    def softmax_with_temperature(self, x, temperature=0.5):
        """Softmax with temperature scaling."""
        exp_values = np.exp(x / temperature)
        return exp_values / np.sum(exp_values)

    def state_to_features(self, state):
        """Convert Taxi-v3 state index to (row, col, passenger, destination)."""
        taxi_row, taxi_col, passenger, destination = self.env.decode(state)
        return np.array([taxi_row, taxi_col, passenger, destination])

    def predict_fn(self, x):
        q_values = []
        for sample in x:
            taxi_row, taxi_col, passenger, destination = map(int, sample)

            # Ensure valid bounds
            taxi_row = min(max(taxi_row, 0), 4)
            taxi_col = min(max(taxi_col, 0), 4)
            passenger = min(max(passenger, 0), 4)
            destination = min(max(destination, 0), 3)

            state = self.env.encode(taxi_row, taxi_col, passenger, destination)

            if state >= self.state_dim or state < 0:
                q_values.append(np.ones(self.action_dim) / self.action_dim)  # Uniform probability
            else:
                q_values.append(self.softmax_with_temperature(self.Q_table[state], temperature=1.0))  # Higher temperature

        return np.array(q_values)  # Shape: (n_samples, action_dim)

    def generate_lime_explanations(self, state_to_explain=62):
        """Generate LIME explanations for the chosen state."""
        explainer = LimeTabularExplainer(
            training_data=np.array([self.state_to_features(s) for s in range(self.state_dim)]),
            feature_names=["taxi_row", "taxi_col", "passenger", "destination"],
            class_names=self.action_labels,  # Action names
            discretize_continuous=False
        )

        explanations = {}
        for action in range(self.action_dim):
            exp = explainer.explain_instance(
                self.state_to_features(state_to_explain),
                self.predict_fn,
                labels=[action],  # Specify the action label for explanation
                num_features=4
            )
            explanations[action] = exp

        return explanations

    def save_lime_explanations(self, explanations, state_to_explain):
        """Save LIME explanations as images in the 'plots' folder."""
        
        for action in range(self.action_dim):
            # Get the LIME explanation figure
            fig = explanations[action].as_pyplot_figure()
            
            # Generate a filename using the state and action
            filename = f"plots/lime_explanation_state_{state_to_explain}_action_{action}_{self.action_symbols[action]}.png"
            
            # Save the figure
            fig.savefig(filename)
            plt.close(fig)  # Close the figure after saving to free up memory
            print(f"Saved explanation for action '{self.action_symbols[action]}' for state {state_to_explain} as {filename}")

    def print_lime_explanations(self, explanations):
        """Print the LIME explanations."""
        
        for action in range(self.action_dim):
            print(f"Explanation for Action {action} ({self.action_symbols[action]}):")
            explanations[action].show_in_notebook()


def get_policy(env, Q_table):
    policy = {}
    states = []
    actions = []

    for state in range(env.observation_space.n):
        best_action = np.argmax(Q_table[state, :])  # Get the action with the highest Q-value
        policy[state] = best_action  # Store the action for that state
        taxi_row, taxi_col, passenger_loc, destination = env.decode(state)
        state_rep = (taxi_row, taxi_col, passenger_loc, destination) 
        states.append(state_rep)
        actions.append(best_action)

    return policy, states, actions

def convert_q_table(agent):
    state_size = agent.env.observation_space.n
    action_size = agent.env.action_space.n
    
    Q_table = np.zeros((state_size, action_size))
    
    for state in range(state_size):
        for action in range(action_size):
            Q_table[state, action] = agent.q_table[state][action]
    
    return Q_table




if __name__ == '__main__':
    
    env = gym.make("Taxi-v3")
    np.set_printoptions(threshold=np.inf) 

    knowledge = rlang.parse_file("./taxi.rlang")
    agent_with_policy = RLangQLearningAgent(env, knowledge=knowledge)
    rewards_with_policy = agent_with_policy.train(episodes=15000)
    
    Q_table = convert_q_table(agent_with_policy)

    policy, states, actions = get_policy(env, Q_table)
    explainability = GetExplainabilityPlotsForEnv(env, Q_table, policy, states, actions)

    explainability.visualize_policy()
    explainability.visualize_decision_tree()
    explainability.plot_feature_importance()
    
    explainability.shap_summary_plot()
    
    explainability.plot_probability_heatmap()
    explainability.plot_logisticregression_probability_heatmap()

    trajectory = explainability.run_episode(Q_table, env)

    explainability.plot_action_trajectory(trajectory)
    explainability.plot_trajectory_state_visits(trajectory)
    explainability.plot_trajectory_rewards(trajectory)

    lime_explainer = LimeExplainer(env, Q_table)

    explanations = lime_explainer.generate_lime_explanations(state_to_explain=62)

    lime_explainer.save_lime_explanations(explanations, state_to_explain=62)
    lime_explainer.print_lime_explanations(explanations)


    """ 
    Lime plot might mean something like:

    taxi_row = 0.11 (not south) suggests that a higher taxi_row makes "South" less likely. For example, if the taxi is higher up in the grid (in a higher row), "South" is less likely to be chosen.

    passenger = 0.08 (south) suggests that having a passenger (or a certain state of the passenger) increases the likelihood of choosing "South".

    taxi_col = 0.04 (not south) suggests that taxi_col has a small negative impact on choosing "South".

    destination = 0.03 (not south) suggests that the destination feature also has a small negative impact on choosing "South
    """