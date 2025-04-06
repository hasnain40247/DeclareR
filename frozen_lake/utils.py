import matplotlib.pyplot as plt
import numpy as np

def plot_training_rewards(rewards, window_size=10,save_path="training_rewards.png"):
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

def plot_comparison_training_rewards(reward_dict, window_size=10, save_path="training_rewards.png"):
    plt.figure(figsize=(10, 5))

    for label, rewards in reward_dict.items():
        episodes = np.arange(len(rewards))
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(episodes, rewards, alpha=0.3, label=f"{label} (raw)")
        plt.plot(episodes[:len(smoothed_rewards)], smoothed_rewards, label=f"{label} (smoothed)")

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Episodes")
    plt.legend()
    plt.grid()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {save_path}")
    plt.close()
