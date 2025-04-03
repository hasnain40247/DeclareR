import tkinter as tk
import gym
from PIL import Image, ImageTk
import numpy as np

class TaxiTkinterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Taxi Environment")
        
        # Create the Taxi environment
        self.env = gym.make("Taxi-v3", render_mode="rgb_array")
        self.observation, self.info = self.env.reset()
        
        # Frame to hold canvas and buttons
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)
        
        # Canvas to display the environment
        self.canvas_width = 500
        self.canvas_height = 500
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(pady=10)
        
        # Control buttons frame
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        # Action buttons
        actions = [("Up", 0), ("Right", 1), ("Down", 2), ("Left", 3), ("Pickup", 4), ("Dropoff", 5)]
        for text, action in actions:
            btn = tk.Button(btn_frame, text=text, width=8, 
                           command=lambda a=action: self.take_action(a))
            btn.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        reset_btn = tk.Button(main_frame, text="Reset Environment", command=self.reset_env)
        reset_btn.pack(pady=10)
        
        # Update the canvas with the initial state
        self.update_canvas()
    
    def update_canvas(self):
        # Get the RGB array from the environment
        rgb_array = self.env.render()
        
        # Convert the RGB array to PIL Image
        img = Image.fromarray(rgb_array)
        
        # Resize the image to fit the canvas
        img = img.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
        
        # Convert to PhotoImage format for Tkinter
        self.photo = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Display current state information
        state_text = f"Current State: {self.observation}"
        self.canvas.create_text(10, 10, text=state_text, anchor=tk.NW, fill="black")
    
    def take_action(self, action):
        # Take the action in the environment
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Update the canvas
        self.update_canvas()
        
        # Check if episode is done
        if terminated or truncated:
            done_text = "Episode finished! Reward: {:.1f}".format(reward)
            self.canvas.create_text(self.canvas_width // 2, self.canvas_height // 2, 
                                   text=done_text, font=("Arial", 20), fill="red")
    
    def reset_env(self):
        # Reset the environment
        self.observation, self.info = self.env.reset()
        self.update_canvas()
    
    def close(self):
        # Close the environment
        self.env.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = TaxiTkinterApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [app.close(), root.destroy()])
    root.mainloop()