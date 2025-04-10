import matplotlib
matplotlib.use = lambda *args, **kwargs: None

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import gym
import rlang
import graphviz
from sklearn.tree import export_graphviz
import scipy.special
import numpy as np
from q_learning import RLangQLearningAgent
from sklearn.preprocessing import LabelEncoder

        
def decode_state(state_index):
    """Decodes the state index into a row and column (in a grid)."""
    grid_width = 12 
    row = state_index // grid_width
    col = state_index % grid_width
    return row, col

class GetExplainabilityPlotsForEnv:
    def __init__(self, env, Q_table, policy, states, actions):
        self.env = env
        self.Q_table = Q_table
        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n
        self.action_labels = ["Up", "Right", "Down", "Left"]
        
        self.policy = policy
        self.states = states
        self.actions = actions
        
        X_train = np.array(states)
        y_train = np.array(actions)
        
        self.X_train = X_train
        self.y_train = y_train
        
        # Initialize and train models
        self.dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=4, random_state=21)
        self.log_reg_model = LogisticRegression(multi_class='ovr')

        self.dt_model.fit(X_train, y_train)
        self.log_reg_model.fit(X_train, y_train)
        
        self.evaluate_models()

    def evaluate_models(self):
        correct_dt_predictions = 0
        correct_lr_predictions = 0

        for state in range(self.state_dim):
            true_action = self.policy[state]
            row, col = self.decode_state(state)  # Get the state representation
            state_rep = (row, col)  # This is your feature vector for the models

            # Predict action using Decision Tree and Logistic Regression
            predicted_action_dt = self.dt_model.predict([state_rep])[0]  # Single prediction, so take [0]
            predicted_action_lr = self.log_reg_model.predict([state_rep])[0]  # Same for Logistic Regression
                            
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

    def decode_state(self, state_index):
        """Decodes the state index into a row and column (in a grid)."""
        # In CliffWalkingEnv, state is just a linearized index, so we need to map it back
        grid_width = 12
        row = state_index // grid_width
        col = state_index % grid_width
        return row, col

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
        
        trajectory["states"].append(next_state)
        trajectory["actions"].append(action)
        _, reward, _, _, _ = env.step(action)
        print(reward)
        trajectory["rewards"].append(reward)

        return trajectory

    def plot_action_trajectory(self, trajectory, filename="plots/episode_cliff_trajectory.png"):
        """Plots the CliffWalking agent's movement across the grid during one episode."""

        states = trajectory["states"]
        coords = []
        grid_rows = 4  # Height of the grid (4 rows)
        grid_cols = 12  # Width of the grid (12 columns)
        # Decode all agent positions (state -> row, col)
        for s in states:
            row, col = s // grid_cols, s % grid_cols
            coords.append((row, col))

        # Define cliff locations (all cells in bottom row except start and goal)
        cliff_locs = [(3, c) for c in range(1, grid_cols - 1)]  # Cliff is in the bottom-most row (3rd row)
        start_loc = (3, 0)  # Start location (bottom-left corner)
        goal_loc = (3, grid_cols - 1)  # Goal location (bottom-right corner)

        fig, ax = plt.subplots(figsize=(grid_cols, grid_rows))

        # Draw grid lines (vertical and horizontal)
        for i in range(grid_rows + 1):
            ax.plot([0, grid_cols], [i, i], color='black', linewidth=1)  # Horizontal grid lines
        for i in range(grid_cols + 1):
            ax.plot([i, i], [0, grid_rows], color='black', linewidth=1)  # Vertical grid lines

        # Draw cliff locations (gray for cliff, in bottom-most row)
        for r, c in cliff_locs:
            ax.add_patch(patches.Rectangle((c, grid_rows - 1 - r), 1, 1, edgecolor='black', facecolor='gray', alpha=0.7))

        # Draw start location (blue for start)
        ax.add_patch(patches.Rectangle((start_loc[1], grid_rows - 1 - start_loc[0]), 1, 1, edgecolor='black', facecolor='blue', alpha=0.5))
        ax.text(start_loc[1] + 0.5, grid_rows - 1 - start_loc[0] + 0.5, "S", ha="center", va="center", fontsize=12, weight='bold', zorder=3)

        # Draw goal location (green for goal)
        ax.add_patch(patches.Rectangle((goal_loc[1], grid_rows - 1 - goal_loc[0]), 1, 1, edgecolor='black', facecolor='green', alpha=0.5))
        ax.text(goal_loc[1] + 0.5, grid_rows - 1 - goal_loc[0] + 0.5, "G", ha="center", va="center", fontsize=12, weight='bold', zorder=3)

        # Draw agent's path (orange circles showing the agent's movement)
        for i, (r, c) in enumerate(coords):
            ax.add_patch(patches.Circle((c + 0.5, grid_rows - 1 - r + 0.5), 0.3, color='orange', edgecolor='black', alpha=0.6, zorder=4))

        # Final formatting
        ax.set_xlim(0, grid_cols)
        ax.set_ylim(0, grid_rows)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("CliffWalking Agent Movement Trajectory")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        
    def plot_trajectory_state_visits(self, trajectory):
        """Plots the state visit heatmap."""
        visits = np.zeros((4, 12))  # CliffWalking grid size is 4x12
        for state in trajectory["states"]:
            row, col = self.decode_state(state)
            visits[row, col] += 1 

        plt.figure(figsize=(6, 6))
        sns.heatmap(visits, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("State Visit Heatmap")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.tight_layout()
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
        plt.tight_layout()
        plt.savefig("plots/episode_rewards_trajectory.png")
        plt.close()

    def visualize_policy(self):
        """Visualizes the learned policy for the CliffWalking environment."""
        
        grid_size = (4, 12)  # CliffWalking grid is 4x12
        policy_grid = np.full(grid_size, " ", dtype="<U2")  # Initialize empty grid

        # Extract optimal actions for each position in the grid
        for state in range(self.state_dim):  # 48 states in CliffWalking
            row, col = self.decode_state(state)
            best_action = np.argmax(self.Q_table[state, :])  # Get best action from Q-table
            policy_grid[row, col] = self.action_labels[best_action]

        plt.figure(figsize=(15, 15))
        plt.imshow(np.zeros(grid_size), cmap="Blues", vmin=-1, vmax=1)
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                plt.text(col, row, policy_grid[row, col], ha='center', va='center', fontsize=14, color='black')

        plt.title("Policy Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("plots/cliffwalking_policy.png")
        plt.close()
        
    def plot_feature_importance(self):
        """Plots feature importance from a Decision Tree model."""
        feature_importance = self.dt_model.feature_importances_
        features = ["row", "col"]

        plt.figure(figsize=(8, 5))
        plt.bar(features, feature_importance, color='skyblue')
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance from Decision Tree")
        plt.tight_layout()
        plt.savefig("plots/feature_importance_decision_tree.png")
        plt.close()

    def shap_summary_plot(self):
        """Generates SHAP summary plot for Logistic Regression model."""
        explainer = shap.KernelExplainer(self.log_reg_model.predict_proba, self.X_train)  # Use the predict_proba method
        shap_values = explainer.shap_values(self.X_train)

        shap.summary_plot(shap_values, self.X_train, feature_names=["row", "col"], class_names=self.action_labels)

    def get_state_representation(self, state_index):
        row, col = self.decode_state(state_index)
        return (row, col)

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

        state_representations = [self.get_state_representation(state) for state in range(self.state_dim)]
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
            row, col = self.decode_state(state)
            state_features = np.array([row, col]).reshape(1, -1)  # Ensure it's a 2D array

            # Get predicted probabilities for all actions
            predicted_probs = self.log_reg_model.predict_proba(state_features)[0]

            # If we get only 3 probabilities, append 0 for the 4th action (left action)
            if len(predicted_probs) == 3:
                predicted_probs = np.append(predicted_probs, 0.0)  # Append 0 for the "left" action

            probabilities.append(predicted_probs)

        # Convert to DataFrame for visualization
        probabilities = np.array(probabilities)
        probabilities_df = pd.DataFrame(probabilities, columns=self.action_labels)

        state_representations = [self.get_state_representation(state) for state in range(self.state_dim)]
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
        plt.savefig("plots/logistic_regression_Q_table_heatmap.png")
        plt.close()
    
    def visualize_decision_tree(self):
        """Visualizes decision tree classifier.""" 
        dot_data = export_graphviz(self.dt_model, 
                                   out_file=None,  
                                   feature_names=["row", "col"],  
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
        self.action_labels = ["Up", "Right", "Down", "Left"]
        self.action_symbols = {
            0: "Up",  
            1: "Right",  
            2: "Down", 
            3: "Left",
        }

        # Define the grid size (assuming a grid of 4 rows and 12 columns for CliffWalking)
        self.grid_rows = 4
        self.grid_cols = 12

    def softmax_with_temperature(self, x, temperature=1.0):
        """Softmax with temperature scaling."""
        exp_values = np.exp(x / temperature)
        return exp_values / np.sum(exp_values)

    def state_to_features(self, state):
        """Convert CliffWalking state index to features (row, col)."""
        row = state // self.grid_cols  # Row is the quotient of state divided by columns
        col = state % self.grid_cols   # Column is the remainder of state divided by columns
        return np.array([row, col])

    def predict_fn(self):
        """Generate prediction function that outputs action probabilities."""
        def predict(x):
            scores = []
            for sample in x:
                # Clamp row and col to ensure they are within the grid bounds
                row, col = map(int, sample)
                row = max(0, min(row, self.grid_rows - 1))  # Ensure row is between 0 and 3
                col = max(0, min(col, self.grid_cols - 1))  # Ensure col is between 0 and 11

                # Convert (row, col) to state index
                state = self.features_to_state(row, col)
                
                # Get the Q-values for the current state
                q_values = self.Q_table[state]
                
                # Apply softmax to get the probabilities for each action
                probabilities = self.softmax_with_temperature(q_values)
                
                # Append the probabilities for the current state
                scores.append(probabilities)
            return np.array(scores)
        return predict

    def features_to_state(self, row, col):
        """Convert (row, col) back to state index."""
        return row * self.grid_cols + col

    def generate_lime_explanations(self, state_to_explain):
        """Generate LIME explanations for the best action at the chosen state."""
        plot_cliff_state(state_to_explain)
        # Get the best action for the state (max Q-value)
        best_action = np.argmax(self.Q_table[state_to_explain])

        # Create the LIME explainer
        explainer = LimeTabularExplainer(
            training_data=np.array([self.state_to_features(s) for s in range(self.state_dim)]),
            feature_names=["row", "col"],  # Features are row and column in the grid
            class_names=self.action_labels,  # Action names are correctly passed
            discretize_continuous=False
        )

        # Generate explanation for only the best action
        exp = explainer.explain_instance(
            self.state_to_features(state_to_explain),
            self.predict_fn(),  # Use the simplified predict_fn
            labels=[best_action],  # Specify the best action label for explanation
            num_features=2  # Now we have 2 features (row and col)
        )
        
        # Print and plot explanations
        self.print_lime_explanations(exp, best_action, state_to_explain)

    def print_lime_explanations(self, explanation, best_action, state_to_explain):
        """Print the LIME explanations and show the plot."""
        print(f"Explanation for State:{state_to_explain} Best Action {best_action} ({self.action_symbols[best_action]}):")
        explanation.show_in_notebook()  # For notebook-based environments

def get_policy(env, Q_table):
    policy = {}
    states = []
    actions = []

    for state in range(env.observation_space.n):
        best_action = np.argmax(Q_table[state, :])  # Get the action with the highest Q-value
        policy[state] = best_action  # Store the action for that state
        
        row, col = decode_state(state)  # Decode the state index into row, col
        state_rep = (row, col)  # State representation as a tuple (row, col)
        
        states.append(state_rep)  # Append the state representation
        actions.append(best_action)  # Append the corresponding action

    return policy, states, actions


def convert_q_table(agent):
    state_size = agent.env.observation_space.n
    action_size = agent.env.action_space.n
    
    Q_table = np.zeros((state_size, action_size))
    
    for state in range(state_size):
        for action in range(action_size):
            Q_table[state, action] = agent.q_table[state][action]
    
    return Q_table


def plot_cliff_state(state, should_save=False):

    grid_rows = 4  # Height of the grid (4 rows)
    grid_cols = 12  # Width of the grid (12 columns)

    # Define cliff locations (all cells in bottom row except start and goal)
    cliff_locs = [(3, c) for c in range(1, grid_cols - 1)]  # Cliff is in the bottom-most row (3rd row)
    start_loc = (3, 0)  # Start location (bottom-left corner)
    goal_loc = (3, grid_cols - 1)  # Goal location (bottom-right corner)

    fig, ax = plt.subplots(figsize=(grid_cols, grid_rows))

    # Draw grid lines (vertical and horizontal)
    for i in range(grid_rows + 1):
        ax.plot([0, grid_cols], [i, i], color='black', linewidth=1)  # Horizontal grid lines
    for i in range(grid_cols + 1):
        ax.plot([i, i], [0, grid_rows], color='black', linewidth=1)  # Vertical grid lines

    # Draw cliff locations (gray for cliff, in bottom-most row)
    for r, c in cliff_locs:
        ax.add_patch(patches.Rectangle((c, grid_rows - 1 - r), 1, 1, edgecolor='black', facecolor='gray', alpha=0.7))

    # Draw start location (blue for start)
    ax.add_patch(patches.Rectangle((start_loc[1], grid_rows - 1 - start_loc[0]), 1, 1, edgecolor='black', facecolor='blue', alpha=0.5))
    ax.text(start_loc[1] + 0.5, grid_rows - 1 - start_loc[0] + 0.5, "S", ha="center", va="center", fontsize=12, weight='bold', zorder=3)

    # Draw goal location (green for goal)
    ax.add_patch(patches.Rectangle((goal_loc[1], grid_rows - 1 - goal_loc[0]), 1, 1, edgecolor='black', facecolor='green', alpha=0.5))
    ax.text(goal_loc[1] + 0.5, grid_rows - 1 - goal_loc[0] + 0.5, "G", ha="center", va="center", fontsize=12, weight='bold', zorder=3)
    
    grid_width = 12
    row = state // grid_width
    col = state % grid_width

    # Drawing the agent's current position with an orange rectangle
    ax.add_patch(patches.Rectangle((col, grid_rows - 1 - row), 1, 1, edgecolor='black', facecolor='orange', alpha=0.5))

    # Add text for the agent's position (optional)
    ax.text(col + 0.5, grid_rows - 1 - row + 0.5, "C", ha="center", va="center", fontsize=12, weight='bold', zorder=3)

    # Final formatting
    ax.set_xlim(0, grid_cols)
    ax.set_ylim(0, grid_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("CliffWalking Current State")

    if not should_save:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig("plots/cliff_state.png")
        plt.close()


if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")
    np.set_printoptions(threshold=np.inf) 

    knowledge = rlang.parse_file("./cliff_walking.rlang")
    agent_with_policy = RLangQLearningAgent(env, knowledge=knowledge,epsilon=0.99 ,epsilon_decay=0.001)
    rewards_with_policy = agent_with_policy.train(episodes=1000)
    print(f"Training complete. Average reward: {agent_with_policy.test(100)}")
    
    Q_table = convert_q_table(agent_with_policy)

    policy, states, actions = get_policy(env, Q_table)
    
    explainability = GetExplainabilityPlotsForEnv(env, Q_table, policy, states, actions)
    explainability.plot_logisticregression_probability_heatmap()

    explainability.visualize_policy()
    explainability.visualize_decision_tree()
    explainability.plot_feature_importance()
    
    explainability.shap_summary_plot()
    
    explainability.plot_probability_heatmap()
    

    trajectory = explainability.run_episode(Q_table, env)

    explainability.plot_action_trajectory(trajectory)
    explainability.plot_trajectory_state_visits(trajectory)
    explainability.plot_trajectory_rewards(trajectory)

    lime_explainer = LimeExplainer(env, Q_table)

    lime_explainer.generate_lime_explanations(state_to_explain=16)
