import customtkinter as ctk
import gymnasium as gym
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import threading
import os
import subprocess
import numpy as np
import json
from PIL import Image, ImageTk
from llm_agents.ReasoningAgent import ReasoningAgent
from reasoningBotFrame import ReasoningBotFrame

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


class RenderScene(ctk.CTkFrame):
    def __init__(self, parent,env_name, algorithm_name):
        super().__init__(parent, fg_color="#FFFDF0")
        self.parent = parent
        self.algorithm_name = algorithm_name
        self.env_name=env_name
        if self.env_name=="taxi":
            self.env = gym.make("Taxi-v3", render_mode="rgb_array")
        elif self.env_name=="frozen_lake":
            self.env = gym.make("FrozenLake-v1", render_mode="rgb_array")
        else:
            self.env = gym.make("CliffWalking-v0", render_mode="rgb_array")



        title=self.env_name[0].upper()+self.env_name[1:]
        self.pack(fill="both", expand=True)
        
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
        
        self.main_frame = ctk.CTkFrame(self, fg_color="#F4F4F4", corner_radius=15)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.control_frame = ctk.CTkFrame(self.main_frame, fg_color="#F4F4F4")
        self.control_frame.pack(fill="x", padx=20, pady=10)
        
        self.reward_data = []  
        
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("Training Progress")
        self.line, = self.ax.plot([], [], label="Reward per Episode")  
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, self.main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.main_frame2 = ctk.CTkFrame(self, fg_color="#F4F4F4", corner_radius=15)
        self.main_frame2.pack(fill="both", expand=True, padx=20, pady=20)
        
        
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

   
        

        self.process = None
        self.running = False
        self.thread = None
        

        self.animation = None
    def _create_param_fields(self):
        """Display hyperparameters in a horizontal row."""
        params = HYPERPARAMETERS.get(self.algorithm_name, {})

        row = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        row.pack(anchor="center", pady=10)



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
            return val  

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
        

        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.canvas.draw()

 
        self.collected_params = {
            key: self._parse_value(entry.get()) for key, entry in self.param_entries.items()
        }
        print("User-defined parameters:", self.collected_params)
        self.running = True
        self.thread = threading.Thread(target=self.run_episodes)
        self.thread.daemon = True
        self.thread.start()
        

        self.animation = FuncAnimation(
            self.fig, 
            self.update_plot, 
            interval=200,  
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
            self.after(0, self.simulation_completed)
    
    def update_plot(self, frame):
        if len(self.reward_data) > 0:
            # Update the line data
            self.line.set_data(range(len(self.reward_data)), self.reward_data)
            
            if hasattr(self, 'avg_line') and len(self.reward_data) >= 10:
                window_size = min(10, len(self.reward_data))
                avg_rewards = np.convolve(
                    self.reward_data, 
                    np.ones(window_size)/window_size, 
                    mode='valid'
                )
                x_avg = range(window_size-1, len(self.reward_data))
                self.avg_line.set_data(x_avg, avg_rewards)
            elif len(self.reward_data) >= 10 and not hasattr(self, 'avg_line'):
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
            
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
        
        return [self.line]

    def test_simulation(self):
        self.row_frame = ctk.CTkFrame(self.main_frame, fg_color="#FFFFFF", corner_radius=10)
        self.row_frame.pack(pady=5, padx=20, fill="both", expand=True)

        # Environment display
        self.display_frame = ctk.CTkFrame(self.row_frame, fg_color="#FFFFFF", corner_radius=10)
        self.display_frame.pack(side="left", padx=20, pady=20, fill="both", expand=True)

        self.display_label = ctk.CTkLabel(self.display_frame, text="")
        self.display_label.pack(pady=10, expand=True)

        # Create a ChatFrame and add it beside the environment display in the row
        agent=ReasoningAgent()
        agent.start_ollama_serve()
        self.chat_frame = ReasoningBotFrame(self.row_frame,agent)
        self.chat_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)

            

        """Run 5 test episodes and display the environment's rgb_array"""
        self.start_button.configure(state="disabled")


        if hasattr(self, 'canvas'):
            widget = self.canvas.get_tk_widget()
            if widget.winfo_exists():
                widget.destroy()
            del self.canvas


        if hasattr(self, 'fig'):
            plt.close(self.fig)  
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
            if self.env_name == "frozen_lake":
                new_width = 400  # Smaller for Frozen Lake
            else:
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
            if episode < 2:
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

