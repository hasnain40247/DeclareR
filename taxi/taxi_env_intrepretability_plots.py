import matplotlib
matplotlib.use = lambda *args, **kwargs: None

import matplotlib.patches as patches
from matplotlib.patches import Circle

import seaborn as sns
import shap
import matplotlib.pyplot as plt
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
    
    def plot_action_trajectory(self, trajectory, filename="plots/episode_taxi_trajectory.png"):
        """Plots the taxi's movement across the grid during one episode using env.locs."""

        states = trajectory["states"]
        coords = []
        locs = {i: tuple(pos) for i, pos in enumerate(self.env.locs)}

        # Decode all taxi positions
        for s in states:
            taxi_row, taxi_col, _, _ = self.env.decode(s)
            coords.append((taxi_row, taxi_col))

        # Decode initial state for passenger and destination
        _, _, passenger_loc, destination = self.env.decode(states[0])

        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw 5x5 grid
        for i in range(6):
            ax.plot([0, 5], [i, i], color='black', linewidth=1)
            ax.plot([i, i], [0, 5], color='black', linewidth=1)

        # Draw special locations (from env.locs)
        colors = ['red', 'green', 'yellow', 'blue']
        for i, (r, c) in locs.items():
            ax.add_patch(patches.Rectangle((c, 4 - r), 1, 1, edgecolor='black',
                                           facecolor=colors[i], alpha=0.3))
            ax.text(c + 0.5, 4 - r + 0.5, '', ha='center', va='center', fontsize=12, weight='bold')
            
        #Draw the taxi on top
        taxi_row, taxi_col, passenger_loc, destination = env.decode(states[0])
        ax.add_patch(patches.Circle((taxi_col + 0.5, 4 - taxi_row + 0.5), 0.3, color='orange', edgecolor='black', zorder=3))
        ax.text(taxi_col + 0.5, 4 - taxi_row + 0.5, 'Taxi', fontsize=10, ha='center', va='center', color='black', weight='bold', zorder=4)

        # Draw passenger
        if passenger_loc < 4:
            pr, pc = locs[passenger_loc]
            ax.text(pc + 0.5, 4 - pr + 0.5, 'P', ha='center', va='center', fontsize=16, color='blue', weight='bold', zorder=4)
        else:
            # Passenger is in the taxi
            taxi_row, taxi_col = coords[0]
            ax.text(taxi_col + 0.5, 4 - taxi_row + 0.8, 'P', ha='center', va='center', fontsize=16, color='blue', weight='bold', zorder=5)

        # Draw destination
        dr, dc = locs[destination]
        ax.text(dc + 0.5, 4 - dr + 0.5, 'D', ha='center', va='center', fontsize=16, color='red', weight='bold', zorder=4)

        # Plot taxi path
        for i, (r, c) in enumerate(coords):
            y_center = 4 - r + 0.5
            x_center = c + 0.5

            # Offset to top-left within cell
            x_topleft = x_center - 0.3
            y_topleft = y_center + 0.3

            ax.add_patch(Circle((x_topleft, y_topleft), 0.08, color='blue', alpha=0.6, zorder=3))



        # Final formatting
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Taxi Movement Trajectory")
        plt.gca().invert_yaxis()
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
        ax.invert_yaxis()  # Flip so row 0 is on top
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

    def softmax_with_temperature(self, x, temperature=0.1):
        """Softmax with temperature scaling."""
        exp_values = np.exp(x / temperature)
        return exp_values / np.sum(exp_values)

    def state_to_features(self, state):
        """Convert Taxi-v3 state index to (row, col, passenger, destination)."""
        taxi_row, taxi_col, passenger, destination = self.env.decode(state)
        return np.array([taxi_row, taxi_col, passenger, destination])

    def predict_fn(self):
        """Generate prediction function that outputs action probabilities."""
        def predict(x):
            scores = []
            for sample in x:
                taxi_row, taxi_col, passenger, destination = map(int, sample)
                taxi_row = int(min(max(taxi_row, 0), 4))
                taxi_col = int(min(max(taxi_col, 0), 4))
                passenger = int(min(max(passenger, 0), 4))
                destination = int(min(max(destination, 0), 3))

                state = self.env.encode(taxi_row, taxi_col, passenger, destination)
                q_values = self.Q_table[state]
                probabilities = self.softmax_with_temperature(q_values)
                scores.append(probabilities)
            return np.array(scores)
        return predict
    
    def generate_lime_explanation_for_chosen_action(self, state_to_explain=62):
        """Generate a LIME explanation for the greedy action at the given state."""
        
        plot_taxi_state(state_to_explain)
        
        # Get the best action for the state (max Q-value)
        best_action = np.argmax(self.Q_table[state_to_explain])

        # Create the LIME explainer
        explainer = LimeTabularExplainer(
            training_data=np.array([self.state_to_features(s) for s in range(self.state_dim)]),
            feature_names=["taxi_row", "taxi_col", "passenger", "destination"], # Features are row and column in the grid
            class_names=self.action_labels,  # Action names are correctly passed
            discretize_continuous=False
        )
        
        explanation = explainer.explain_instance(
            self.state_to_features(state_to_explain),
            self.predict_fn(),
            labels=[best_action], 
            num_features=4
        )
                
        print(f"Env locations for State {env.locs}:")

        # Show feature-wise contributions for the chosen action
        print(f"\nExplanation for State:{state_to_explain} Action {best_action} ({self.action_symbols[best_action]}):")
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

def plot_taxi_state(state, should_save = False):
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
        plt.gca().invert_yaxis()
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

    lime_explainer.generate_lime_explanation_for_chosen_action(state_to_explain=1)
