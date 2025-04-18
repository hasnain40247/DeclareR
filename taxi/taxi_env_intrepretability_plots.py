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
import matplotlib.patches as patches
from matplotlib.patches import Circle

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
        self.log_reg_model = LogisticRegression(max_iter=500)

        self.dt_model.fit(X_train, y_train)
        self.log_reg_model.fit(X_train, y_train)
        
        # Evaluate how well the models are imitating the Q-learning agent
        self.evaluate_models()

    def evaluate_models(self):
        correct_dt_predictions = 0
        correct_lr_predictions = 0

        for state in range(self.state_dim):
            true_action = self.policy[state]  # True action taken by Q-learning agent
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

    def plot_taxi_trajectory(self, trajectory, wall_locations, filename="plots/episode_taxi_trajectory.png"):
        """Plots the taxi's movement across the grid during one episode using env.locs and wall data."""

        states = trajectory["states"]
        coords = []
        locs = {i: tuple(pos) for i, pos in enumerate(self.env.locs)}

        for s in states:
            taxi_row, taxi_col, _, _ = self.env.decode(s)
            coords.append((taxi_row, taxi_col))

        _, _, passenger_loc, destination = self.env.decode(states[0])

        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw 5x5 grid
        for i in range(6):
            ax.plot([0, 5], [i, i], color='black', linewidth=1)
            ax.plot([i, i], [0, 5], color='black', linewidth=1)

        # Draw special locations (R, G, Y, B)
        colors = ['red', 'green', 'yellow', 'blue']
        for i, (r, c) in locs.items():
            ax.add_patch(patches.Rectangle((c, 4 - r), 1, 1, edgecolor='black',
                                           facecolor=colors[i], alpha=0.3))

        # Draw walls
        for (r, c), direction in wall_locations:
            x = c
            y = 4 - r  # Flip row for plotting

            if direction == "east":
                ax.plot([x + 1, x + 1], [y, y + 1], color='black', linewidth=4)
            elif direction == "west":
                ax.plot([x, x], [y, y + 1], color='black', linewidth=4)
            elif direction == "north":
                ax.plot([x, x + 1], [y + 1, y + 1], color='black', linewidth=4)
            elif direction == "south":
                ax.plot([x, x + 1], [y, y], color='black', linewidth=4)

        # Draw initial taxi position
        taxi_row, taxi_col, _, _ = self.env.decode(states[0])
        ax.add_patch(patches.Circle((taxi_col + 0.5, 4 - taxi_row + 0.5), 0.3, color='orange', edgecolor='black', zorder=3))
        ax.text(taxi_col + 0.5, 4 - taxi_row + 0.5, 'Taxi', fontsize=10, ha='center', va='center', color='black', weight='bold', zorder=4)

        # Draw passenger
        if passenger_loc < 4:
            pr, pc = locs[passenger_loc]
            ax.text(pc + 0.5, 4 - pr + 0.5, 'P', ha='center', va='center', fontsize=16, color='blue', weight='bold', zorder=4)
        else:
            taxi_row, taxi_col = coords[0]
            ax.text(taxi_col + 0.5, 4 - taxi_row + 0.8, 'P', ha='center', va='center', fontsize=16, color='blue', weight='bold', zorder=5)

        # Draw destination
        dr, dc = locs[destination]
        ax.text(dc + 0.5, 4 - dr + 0.5, 'D', ha='center', va='center', fontsize=16, color='red', weight='bold', zorder=4)

        # Plot taxi path
        for i, (r, c) in enumerate(coords):
            y_center = 4 - r + 0.5
            x_center = c + 0.5
            x_topleft = x_center - 0.3
            y_topleft = y_center + 0.3
            ax.add_patch(Circle((x_topleft, y_topleft), 0.08, color='blue', alpha=0.6, zorder=3))

        # Final formatting
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Taxi Movement Trajectory")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()





    def plot_trajectory_state_visits(self, trajectory):
        """Plots the state visit heatmap."""
        visits = np.zeros((self.grid_size, self.grid_size))
        for state in trajectory["states"]:
            taxi_row, taxi_col, _, _ = self.env.decode(state)
            visits[taxi_row, taxi_col] += 1 

        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(visits, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("State Visit Heatmap")
        plt.xlabel("Taxi Column")
        plt.ylabel("Taxi Row")
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
        """Visualizes the learned policy for the Taxi-v3 environment."""

        grid_size = 5  # Taxi-v3 environment grid is 5x5
        policy_grid = np.full((grid_size, grid_size), " ", dtype="<U2")  # Initialize empty grid

        # Extract optimal actions for each taxi position
        for state in range(self.state_dim):
            taxi_row, taxi_col, pass_loc, dest_loc = self.env.decode(state)
            best_action = np.argmax(self.Q_table[state, :])
            policy_grid[taxi_row, taxi_col] = self.action_symbols[best_action]

        plt.figure(figsize=(6, 6))
        plt.imshow(np.zeros((grid_size, grid_size)), cmap="Blues", vmin=-1, vmax=1)

        # Display action symbols on the grid
        for row in range(grid_size):
            for col in range(grid_size):
                plt.text(col, row, policy_grid[row, col], ha='center', va='center', fontsize=14, color='black')

        plt.title("Policy Visualization")
        plt.axis('off')
        plt.tight_layout()
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
        plt.tight_layout()
        plt.savefig("plots/feature_importance_decision_tree.png")
        plt.close()

    def shap_summary_plot(self):
        """Generates SHAP summary plot for Logistic Regression model."""
        explainer = shap.KernelExplainer(self.log_reg_model.predict_proba, self.X_train)  # Use the predict_proba method
        shap_values = explainer.shap_values(self.X_train)

        shap.summary_plot(shap_values, self.X_train, feature_names=["taxi_row", "taxi_col", "passenger_loc", "destination"], class_names=self.action_labels)

    
    def get_state_representation(self, state_index):
        taxi_row, taxi_col, passenger_loc, destination = self.env.decode(state_index)
        return (taxi_row, taxi_col, env.locs[passenger_loc], env.locs[destination])

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
    def __init__(self, env, Q_table, walls):
        self.env = env
        self.walls = walls
        self.Q_table = Q_table
        self.state_dim = 500  # assuming a predefined state space dimension
        self.action_labels = ["north", "south", "east", "west", "pickup", "dropoff"]

    def state_to_features(self, state):
        """ Convert state to features (example implementation) """
        taxi_row, taxi_col, passenger, destination = self.env.decode(state)
        return [taxi_row, taxi_col, passenger, destination]
    
    def softmax_with_temperature(self, x, temperature=1.0):
        """Softmax with temperature scaling."""
        exp_values = np.exp(x / temperature)
        return exp_values / np.sum(exp_values)


    def predict_fn(self, epsilon=0.1):
        """Generate prediction function that outputs action probabilities using epsilon-greedy policy."""

        def bin_feature(value, bins):
            """Assign a value to a specific bin."""
            for i in range(len(bins) - 1):
                if bins[i] <= value < bins[i+1]:
                    return i
            return len(bins) - 1  # If the value is in the last bin

        def predict(x):
            scores = []

            # Define your custom bins for each feature (adjust as per your requirement)
            taxi_row_bins = [0, 1, 2, 3, 4]  # 0-1 -> bin 1, 1-2 -> bin 2, etc.
            taxi_col_bins = [0, 1, 2, 3, 4]
            passenger_bins = [0, 1, 2, 3, 4]
            destination_bins = [0, 1, 2, 3]

            for sample in x:
                taxi_row, taxi_col, passenger, destination = map(int, sample)

                # Discretize each feature using the defined bins
                taxi_row = bin_feature(taxi_row, taxi_row_bins)
                taxi_col = bin_feature(taxi_col, taxi_col_bins)
                passenger = bin_feature(passenger, passenger_bins)
                destination = bin_feature(destination, destination_bins)

                # Encode the state using the environment
                state = self.env.encode(taxi_row, taxi_col, passenger, destination)

                # Fetch the Q-values for the state (in case you want to compute the probabilities)
                q_values = self.Q_table[state]

                # Apply softmax to get action probabilities (if needed)
                probabilities = self.softmax_with_temperature(q_values)

                scores.append(probabilities)

            return np.array(scores)  # Only return the action probabilities (no tuple)

        return predict

    def generate_lime_explanation_for_chosen_action(self, state_to_explain=62):
        """ Generate a LIME explanation for the greedy action at the given state """
        
        plot_taxi_state(state_to_explain, self.walls)
        
        # Get the best action for the state (max Q-value)
        best_action = np.argmax(self.Q_table[state_to_explain])

        # Define custom bin edges for each feature
        bin_edges = {
            'taxi_row': [0, 1, 2, 3, 4],  # bins 0, 1, 2, 3
            'taxi_col': [0, 1, 2, 3, 4],  # bins 0, 1, 2, 3
            'passenger': [0, 1, 2, 3, 4],  # bins 0, 1, 2, 3
            'destination': [0, 1, 2, 3]    # bins 0, 1, 2
        }

        # Create the LIME explainer with custom binning
        explainer = LimeTabularExplainer(
            training_data=np.array([self.state_to_features(s) for s in range(self.state_dim)]),
            feature_names=["taxi_row", "taxi_col", "passenger", "destination"],
            class_names=self.action_labels,
            discretize_continuous=False,  # Enable binning
        )
        
        # Get the predict function
        predict_function = self.predict_fn()

        # Ensure input is a 2D array (reshape if needed)
        state_features = np.array([self.state_to_features(state_to_explain)])
        
        # Make sure to modify the predict_fn to handle epsilon-greedy with proper discretization
        explanation = explainer.explain_instance(
            state_features[0],
            predict_function,
            labels=[best_action],
            num_features=4,
            num_samples=1000
        )

                
        print(f"Env locations for State {self.env.locs}:")
        print(f"\nExplanation for State:{state_to_explain} Action {best_action} ({self.action_labels[best_action]}):")
        explanation.show_in_notebook()
        

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

def get_walls():
    wall_locations = []
    for row in range(5):
        for col in range(5):
            for action, direction in enumerate(['south', 'north', 'east', 'west']):
                state = env.encode(row, col, 0, 1)  # arbitrary passenger/dest
                transitions = env.P[state][action]
                for prob, next_state, _, _ in transitions:
                    next_row, next_col, _, _ = env.decode(next_state)
                    if (next_row, next_col) == (row, col):  # no movement => obstacle
                        wall_locations.append(((row, col), direction))

    return wall_locations

def plot_taxi_state(state, walls, should_save = False):
        locs = {
            0: (0, 0),  # R
            1: (0, 4),  # G
            2: (4, 0),  # Y
            3: (4, 3),  # B
        }
        
        taxi_row, taxi_col, passenger_loc, destination = env.decode(state)

        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw grid
        for i in range(6):
            ax.plot([0, 5], [i, i], color='black', linewidth=1)
            ax.plot([i, i], [0, 5], color='black', linewidth=1)

        # Draw special locations
        label_map = {0: 'R', 1: 'G', 2: 'Y', 3: 'B'}
        colors = {'R': 'red', 'G': 'green', 'Y': 'yellow', 'B': 'blue'}

        for loc, (r, c) in locs.items():
            label = label_map[loc]
            ax.add_patch(patches.Rectangle((c, 4 - r), 1, 1, edgecolor='black',
                                           facecolor=colors[label], alpha=0.3))
            ax.text(c + 0.5, 4 - r + 0.5, "", ha='center', va='center', fontsize=12, weight='bold', zorder=2)
        
        # Draw walls
        for (r, c), direction in walls:
            x = c
            y = 4 - r  # Flip row for plotting

            if direction == "east":
                ax.plot([x + 1, x + 1], [y, y + 1], color='black', linewidth=4)
            elif direction == "west":
                ax.plot([x, x], [y, y + 1], color='black', linewidth=4)
            elif direction == "north":
                ax.plot([x, x + 1], [y + 1, y + 1], color='black', linewidth=4)
            elif direction == "south":
                ax.plot([x, x + 1], [y, y], color='black', linewidth=4)


        # Draw the taxi on top
        ax.add_patch(patches.Circle((taxi_col + 0.5, 4 - taxi_row + 0.5), 0.3, color='orange', edgecolor='black', zorder=3))
        ax.text(taxi_col + 0.5, 4 - taxi_row + 0.5, 'Taxi', fontsize=10, ha='center', va='center', color='black', weight='bold', zorder=4)

        # Draw passenger
        if passenger_loc < 4:
            pr, pc = locs[passenger_loc]
            ax.text(pc + 0.5, 4-pr + 0.5, 'P', ha='center', va='center', fontsize=16, color='blue', weight='bold', zorder=4)
        else:
            # Passenger is in the taxi
            ax.text(taxi_col + 0.5, 4 - taxi_row + 0.8, 'P in taxi', ha='center', va='center', fontsize=16, color='blue', weight='bold', zorder=5)

        # Draw destination
        dr, dc = locs[destination]
        ax.text(dc + 0.5, 4 - dr + 0.5, 'D', ha='center', va='center', fontsize=16, color='red', weight='bold', zorder=4)

       
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Taxi-v3 Environment Initial State", fontsize=14)
        plt.grid(False)
        
        if not should_save:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig("plots/initial_env_state.png")
            plt.close()
            
if __name__ == '__main__':
    
    env = gym.make("Taxi-v3")
    np.set_printoptions(threshold=np.inf) 

    knowledge = rlang.parse_file("./taxi.rlang")
    agent_with_policy = RLangQLearningAgent(env, knowledge=knowledge)
    rewards_with_policy = agent_with_policy.train(episodes=15000)
    
    Q_table = convert_q_table(agent_with_policy)
    walls = get_walls()

    policy, states, actions = get_policy(env, Q_table)
    explainability = GetExplainabilityPlotsForEnv(env, Q_table, policy, states, actions)

    explainability.visualize_policy()
    explainability.visualize_decision_tree()
    explainability.plot_feature_importance()
    
    explainability.shap_summary_plot()
    
    explainability.plot_probability_heatmap()
    explainability.plot_logisticregression_probability_heatmap()

    trajectory = explainability.run_episode(Q_table, env)

    explainability.plot_taxi_trajectory(trajectory, walls)
    explainability.plot_trajectory_state_visits(trajectory)
    explainability.plot_trajectory_rewards(trajectory)

    lime_explainer = LimeExplainer(env, Q_table, walls)

    lime_explainer.generate_lime_explanation_for_chosen_action(state_to_explain=342)
