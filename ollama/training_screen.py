import subprocess
import customtkinter as ctk
import gymnasium as gym
import numpy as np
import time
from PIL import Image, ImageTk
import threading
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

HYPERPARAMETERS = {
    "Q-Learning": {
        "alpha": 0.9,
        "gamma": 0.9,
        "epsilon": 1,
        "epsilon_decay": 0.0001,
        "knowledge": None

    },
    "Dyna-Q": {
        "n_planning_steps": 5,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.1,
        "knowledge": None,
  

        "p_policy": 0.2
    },
    "R-Max": {
        "r_max": 20,
        "gamma": 0.95,
        "delta": 0.01,
        "M": 1,
        "knowledge": None

    }
}

class EnvRenderer(ctk.CTkFrame):
    def __init__(self, parent,env_name, algorithm_name):
        super().__init__(parent, fg_color="#FFFDF0")
        self.parent = parent
        self.algorithm_name = algorithm_name
        self.env_name=env_name
        # Create the gym environment
        if self.env_name=="taxi":
            self.env = gym.make("Taxi-v3", render_mode="rgb_array")
        else:
            self.env = gym.make("CliffWalking-v0", render_mode="rgb_array")



        title=self.env_name[0].upper()+self.env_name[1:]
        # Configure frame
        self.pack(fill="both", expand=True)
        
        # Title with selected algorithm
        self.title_label = ctk.CTkLabel(
            self, 
            text=f"{title} Environment - {self.algorithm_name}", 
            font=("Inter", 28, "bold"),
            text_color="#424242"
        )
        self.title_label.pack(pady=(20, 10))
        self.param_entries = {}
        self.param_frame = ctk.CTkFrame(self, fg_color="#FFFDF0")
        self.param_frame.pack(pady=10, padx=20, fill="x")

        self._create_param_fields()
        
        # Main frame
        self.main_frame = ctk.CTkFrame(self, fg_color="#F4F4F4", corner_radius=15)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.control_frame = ctk.CTkFrame(self.main_frame, fg_color="#F4F4F4")
        self.control_frame.pack(fill="x", padx=20, pady=10)
        
        self.reward_data = []  # To store episodic rewards for plotting
        
        # Initialize the plot (only for training, not for testing)
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("Training Progress")
        self.line, = self.ax.plot([], [], label="Reward per Episode")  # Empty line at first
        self.ax.legend()

        # Embed plot into Tkinter window (this will be replaced later)
        self.canvas = FigureCanvasTkAgg(self.fig, self.main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Main frame
        self.main_frame2 = ctk.CTkFrame(self, fg_color="#F4F4F4", corner_radius=15)
        self.main_frame2.pack(fill="both", expand=True, padx=20, pady=20)
        
        
        # Add status label
        self.status_label = ctk.CTkLabel(
            self.main_frame2,
            text="Ready to start simulation",
            font=("Inter", 24, "bold"),
            text_color="#424242"
        )
        self.status_label.pack(pady=(5, 10))
        
        # Control buttons
        self.start_button = ctk.CTkButton(
            self.control_frame, 
            text="Train", 
            command=self.start_simulation,
            fg_color="#FFD3B6",
            text_color="#424242",
            font=("Inter", 16, "bold"),
            corner_radius=10,
            width=200,  
            height=50,
            hover_color="#FFC59E"
        )
        self.start_button.pack(side="left", padx=10)

        self.test_button = ctk.CTkButton(
            self.control_frame, 
            text="Test", 
            command=self.test_simulation,
            fg_color="#FFD3B6",
            text_color="#424242",
            font=("Inter", 16, "bold"),
            corner_radius=10,
            width=200,  
            height=50,
            hover_color="#FFC59E"
        )
        self.test_button.pack(side="left", padx=10)

        # Back button
        self.back_button = ctk.CTkButton(
            self.control_frame, 
            text="Back to Selection", 
            command=self.back_to_selection,
            fg_color="#A8D8EA",
            text_color="#424242",
            font=("Inter", 16, "bold"),
            corner_radius=10,
            width=200,  
            height=50,
            hover_color="#95C9DB"
        )
        self.back_button.pack(side="right", padx=10)

   
        
        # Process and thread variables
        self.process = None
        self.running = False
        self.thread = None
        
        # Animation related
        self.animation = None
    def _create_param_fields(self):
        """Display hyperparameters in a horizontal row."""
        params = HYPERPARAMETERS.get(self.algorithm_name, {})

        row = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        row.pack(anchor="center", pady=10)

        # row.pack(fill="x", pady=10)

        for key, value in params.items():
            field_frame = ctk.CTkFrame(row, fg_color="transparent")
            field_frame.pack(side="left", padx=10)

            label = ctk.CTkLabel(field_frame, text=key, font=("Inter", 18,"bold"))
            label.pack()

            entry = ctk.CTkEntry(field_frame, width=90)
            entry.insert(0, str(value))
            entry.pack()

            self.param_entries[key] = entry

    def _parse_value(self, val):
        try:
            if '.' in val:
                return float(val)
            else:
                return int(val)
        except ValueError:
            return val  # fallback to string if not numeric

    def start_simulation(self):
        if self.running:
            return
            
        self.clear_test_display()
        if not hasattr(self, 'canvas') or not self.canvas.get_tk_widget().winfo_exists():

          
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.ax.set_xlabel("Episodes")
            self.ax.set_ylabel("Reward")
            self.ax.set_title("Training Progress")
            self.line, = self.ax.plot([], [], label="Reward per Episode")  # Empty line at first
            self.ax.legend()
            self.canvas = FigureCanvasTkAgg(self.fig, self.main_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)


        self.start_button.configure(state="disabled")
        self.status_label.configure(text="Simulation running...")
        self.reward_data = []  # Reset rewards
        
        # Reset plot
        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.canvas.draw()

        # Start the simulation in a separate thread
        self.collected_params = {
            key: self._parse_value(entry.get()) for key, entry in self.param_entries.items()
        }
        print("User-defined parameters:", self.collected_params)
        self.running = True
        self.thread = threading.Thread(target=self.run_episodes)
        self.thread.daemon = True
        self.thread.start()
        
        # Start the animation for smooth updates
        self.animation = FuncAnimation(
            self.fig, 
            self.update_plot, 
            interval=200,  # Update every 200ms
            blit=False
        )
        self.canvas.draw()

    def run_episodes(self):
        try:
            python_interpreter = '../../.venv/bin/python'  # On Linux/macOS
            script_path = f'../selector.py'
            os.chdir(self.env_name)
           
          
            filename = f"./{self.env_name}/{self.env_name}.rlang"
            hyperparams_str = json.dumps(self.collected_params)


            args = [python_interpreter, script_path, filename,self.env_name,self.algorithm_name,hyperparams_str]
         
            # result = subprocess.run(
            #         args=args,
            #         capture_output=True,
            #         text=True
            #     )

            # output = result.stdout.strip()
            # error_output = result.stderr.strip()
            # print(output)
            # print(error_output)
            self.process = subprocess.Popen(
                args, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                bufsize=1  # Line buffered
            )
            # Process output in real-time
            for line in iter(self.process.stdout.readline, ''):
                print(line)
             
                if not self.running:
                    break
                    
                if line.startswith("Episode"):
                    try:
                        reward = int(line.split(":")[2].strip())
                        self.reward_data.append(reward)
                        # Update status with latest episode
                        episode_num = len(self.reward_data)
                        # Use after() to safely update the UI from a non-main thread
                        self.after(0, lambda: self.status_label.configure(
                            text=f"Running episode {episode_num} |  Reward = {reward}"
                        ))
                    except (IndexError, ValueError) as e:
                        print(f"Error parsing line: {line}, Error: {e}")
            
            self.process.stdout.close()
            self.process.wait()
            print(os.getcwd())
           
            with open(f"./training_details.json","r") as f:
                self.q_table=json.load(f)[-1]["q_table"]
      
            os.chdir("..")
            
        except Exception as e:
            print(f"Error in run_episodes: {e}")
        finally:
            self.running = False
            # Use after() to safely update UI from a non-main thread
            self.after(0, self.simulation_completed)
    
    def update_plot(self, frame):
        """Called by FuncAnimation to update the plot smoothly"""
        if len(self.reward_data) > 0:
            # Update the line data
            self.line.set_data(range(len(self.reward_data)), self.reward_data)
            
            # Add smoothed moving average line if enough data points
            if hasattr(self, 'avg_line') and len(self.reward_data) >= 10:
                # Simple moving average for smoothing
                window_size = min(10, len(self.reward_data))
                avg_rewards = np.convolve(
                    self.reward_data, 
                    np.ones(window_size)/window_size, 
                    mode='valid'
                )
                x_avg = range(window_size-1, len(self.reward_data))
                self.avg_line.set_data(x_avg, avg_rewards)
            elif len(self.reward_data) >= 10 and not hasattr(self, 'avg_line'):
                # Create the average line on first run with enough data
                window_size = 10
                avg_rewards = np.convolve(
                    self.reward_data, 
                    np.ones(window_size)/window_size, 
                    mode='valid'
                )
                x_avg = range(window_size-1, len(self.reward_data))
                self.avg_line, = self.ax.plot(
                    x_avg, avg_rewards, 
                    'r-', 
                    label="Moving Average (10 episodes)"
                )
                self.ax.legend()
            
            # Adjust limits if needed
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
        
        return [self.line]

    def test_simulation(self):
               # Environment display
        self.display_frame = ctk.CTkFrame(self.main_frame, fg_color="#FFFFFF", corner_radius=10)
        self.display_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        self.display_label = ctk.CTkLabel(self.display_frame, text="")
        self.display_label.pack(pady=10, expand=True)
        
 
        # self.status_label.pack(pady=(5, 10))
        
        """Run 5 test episodes and display the environment's rgb_array"""
        self.start_button.configure(state="disabled")
        # self.status_label.configure(text="Running Test Episodes...")

        if hasattr(self, 'canvas'):
            widget = self.canvas.get_tk_widget()
            if widget.winfo_exists():
                widget.destroy()
            del self.canvas

        # Clear figure and axes references
        if hasattr(self, 'fig'):
            plt.close(self.fig)  # Properly close the matplotlib figure
            del self.fig
        if hasattr(self, 'ax'):
            del self.ax
        if hasattr(self, 'line'):
            del self.line
        if hasattr(self, 'avg_line'):
            del self.avg_line


        self.start_button.configure(state="disabled")
        # self.status_label.configure(text=f"Status: Running {self.algorithm_name} Simulation")
        
        # Schedule the episode runner
        self.after(10, self.run_episode, 1, 0)

            
    def update_display(self, frame):
            # Convert numpy array to PIL Image
            img = Image.fromarray(frame)
            
            # Resize image to fit display better
            width, height = img.size
            new_width = 800
            new_height = int(height * (new_width / width))
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.display_label.configure(image=photo)
            self.display_label.image = photo
            
 

    def run_episode(self, episode, total_reward_sum):
        self.status_label.configure(text=f"Running Test Episode {episode}/5")

        # Reset environment
        observation, info = self.env.reset()
        done = False
        terminated = False
        truncated = False
        step = 0
        episode_reward = 0

        self.run_step(episode, step, observation, episode_reward, total_reward_sum, done, terminated, truncated)
        

    def test_completed(self):
        self.start_button.configure(state="normal")
        self.status_label.configure(text="Test completed! You can train again or go back.")
    def clear_test_display(self):
        if hasattr(self, 'display_frame') and self.display_frame:
            self.display_frame.destroy()
            del self.display_frame

        if hasattr(self, 'display_label') and self.display_label:
            self.display_label.destroy()
            del self.display_label

    def run_step(self, episode, step, observation, episode_reward, total_reward_sum, done, terminated, truncated):
        if done:
            total_reward_sum += episode_reward
            if episode < 5:
                self.after(1000, self.run_episode, episode + 1, total_reward_sum)
            else:
                # All test episodes done, update UI
                self.after(0, self.test_completed)
            return



        state = observation
      
        action = np.argmax(self.q_table[state])  # Choose action with max Q-value
       
        observation, reward, terminated, truncated, info = self.env.step(action)

        step += 1
        episode_reward += reward
        self.update_display(self.env.render())

        done = terminated or truncated
        self.after(100, self.run_step, episode, step, observation, episode_reward, total_reward_sum, done, terminated, truncated)  
    def display_image(self, img_tk):
        """Display the rendered image of the environment"""
        label = ctk.CTkLabel(self.main_frame, image=img_tk)
        label.image = img_tk  # Keep a reference to avoid garbage collection
        label.pack(fill="both", expand=True)
        
    def simulation_completed(self):
        """Called when simulation is done"""
        if hasattr(self, 'animation') and self.animation:
            self.animation.event_source.stop()
            self.animation = None
            
        self.start_button.configure(state="normal")
        self.status_label.configure(text="Simulation completed!")

    def back_to_selection(self):
        # Stop any running simulation
        if self.running:
            self.running = False
            if self.process:
                try:
                    self.process.terminate()
                except:
                    pass
                    
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.0)
        
        if hasattr(self, 'animation') and self.animation:
            self.animation.event_source.stop()
            
        self.env.close()
        self.destroy()
        self.parent.show_selection_screen()

        
    def __del__(self):
        """Cleanup when the object is destroyed"""
        if hasattr(self, 'running') and self.running:
            self.running = False
            if hasattr(self, 'process') and self.process:
                try:
                    self.process.terminate()
                except:
                    pass
class SelectAlgorithmScreen(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="#FFFDF0")
        self.parent = parent
        
        # Configure frame
        self.pack(fill="both", expand=True)
        
        title = ctk.CTkLabel(
            self, 
            text="Choose an Algorithm", 
            font=("Inter", 28, "bold"), 
            text_color="#424242"
        )
        title.pack(pady=(30, 10))
        
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(expand=True, fill="both", padx=50, pady=20)

        # Grid setup (set only one column for vertical alignment)
        container.grid_columnconfigure(0, weight=1)

        # Q-Learning
        self._create_algorithm_card(
            parent=container,
            column=0,
            name="Q-Learning",
            callback=lambda: self.parent.launch_algorithm("Q-Learning")
        )

        # Dyna-Q
        self._create_algorithm_card(
            parent=container,
            column=0,
            name="Dyna-Q",
            callback=lambda: self.parent.launch_algorithm("Dyna-Q")
        )

        # R-Max
        self._create_algorithm_card(
            parent=container,
            column=0,
            name="R-Max",
            callback=lambda: self.parent.launch_algorithm("R-Max")
        )

    def _create_algorithm_card(self, parent, column, name, callback):
        frame = ctk.CTkFrame(parent, corner_radius=15, fg_color="#F4F4F4")
        
        label = ctk.CTkLabel(
            frame, 
            text=name, 
            font=("Inter", 22, "bold"), 
            text_color="#333"
        )
        label.pack(pady=(15, 10))

        button = ctk.CTkButton(
            frame,
            text=f"Use {name}",
            fg_color="#FFD3B6",
            text_color="#424242",
            font=("Inter", 18, "bold"),
            command=callback,
            corner_radius=10,
            hover_color="#FFC59E"
        )
        button.pack(pady=(0, 15))
        
        # Get the next available row
        frame.grid(row=parent.grid_size()[1], column=column, padx=20, pady=20, sticky="nsew")


class MainApplication(ctk.CTk):
    def __init__(self,env_name):
        super().__init__()
        self.title("Algorithm Playground")
        self.attributes("-fullscreen", True)
        self.configure(fg_color="#FFFDF0")
        self.env_name=env_name
        ctk.set_appearance_mode("light")
        
   
        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))
        
   
        self.show_selection_screen()
    
    def show_selection_screen(self):
    
        for widget in self.winfo_children():
            widget.destroy()
        

        self.selection_screen = SelectAlgorithmScreen(self)
    
    def launch_algorithm(self, algorithm_name):

        for widget in self.winfo_children():
            widget.destroy()
    
        self.renderer = EnvRenderer(self,self.env_name, algorithm_name)


if __name__ == "__main__":
    app = MainApplication(env_name="cliff_walking")
    app.mainloop()


