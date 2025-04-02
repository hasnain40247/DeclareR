import time
import customtkinter as ctk
import threading
import constants
from EffectAgent import EffectAgent
from PolicyAgent import PolicyAgent
import re
import os


def extract_policy_name(policy_text):
    match = re.search(r"Policy\s+(\w+):", policy_text)
    return match.group(1) if match else "unknown_policy"

class ChatApp(ctk.CTk):
    def __init__(self,effect_agent, policy_agent, environment_constants, vocab=None,env_name="taxi"):
        super().__init__()
        self.generated_effect_history = [] 
        self.env_name=env_name
        
        self.generated_policy_history=[]
        self.vocab=vocab


        self.effect_agent = effect_agent
        self.policy_agent = policy_agent
        self.effect_text = None
        self.policy_text = None
        self.environment_constants = environment_constants
   
        self.configure(bg="#86A788") 
        self.configure(fg_color="#FFFDF0") 

        ctk.set_appearance_mode("dark")  

        self.title("Ollama Taxi RLang Policy Generator")
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

     
        self.chat_container = ctk.CTkFrame(
            self,
            fg_color="transparent",  
            border_width=3,  
            border_color="#FFD3B6",
            corner_radius=25  
        )
        self.chat_container.grid(row=0, column=0, padx=40, pady=20, columnspan=2, sticky="nsew")

    
        self.chat_frame = ctk.CTkScrollableFrame(
            self.chat_container, 
            width=1200, 
            height=700, 
            fg_color="#FFFDF0",  
            corner_radius=25  
        )
        self.chat_frame.pack(expand=True, fill="both", padx=10, pady=10)
              
        self.welcome_message = ctk.CTkLabel(
            self.chat_frame,
            text="Welcome to the DeclareR Plan Generator!\nStart by typing your advice below.",
            wraplength=800,
            justify="center",
            fg_color="transparent",
            text_color="#424242",
            font=("Inter", 27, "italic")
        )
        self.welcome_message.pack(expand=True, pady=150)  


        self.input_container = ctk.CTkFrame(
            self,
            fg_color="#EFF3EA",  # White background
            corner_radius=25,  # Rounded container
            border_width=2,
            border_color="#D9DFC6",  # Pastel Mint border
        )
        self.input_container.grid(row=1, column=0, padx=40, pady=10, sticky="ew",columnspan=2)  # Stretch full width)
        self.input_container.grid_columnconfigure(0, weight=1)
        self.input_container.grid_columnconfigure(1, weight=0)

      
        self.user_input = ctk.CTkEntry(
            self.input_container,
            placeholder_text="Describe your Effect...",
            fg_color="transparent",  
            text_color="#424242",
            font=("Inter", 22, "normal"),
            border_width=0,  
        )
        self.user_input.grid(row=0, column=0, sticky="ew", padx=(15, 5), pady=15)

 
        self.send_button = ctk.CTkButton(
            self.input_container,
            text="↵",
            width=45,
            height=45,
            fg_color="#D9DFC6",
            text_color="#424242",
            font=("Inter", 20, "bold"),
            corner_radius=100,
            hover=False,
            command=self.generate_output
        )
        self.send_button.grid(row=0, column=1, padx=(5, 10), pady=5)


        self.is_generating_effect = True
       
       
        
        self.toggle_button = ctk.CTkButton(
            self,
            text="Switch To Policy Mode",
            fg_color="#FFD3B6",
            text_color="#424242",
            border_width=3,
            font=("Inter", 20, "bold"),
            height=60,
            hover=False,
            border_color="#FFD3B6",  
            command=self.toggle_mode
        )
        self.toggle_button.grid(row=2, column=0,pady=10, padx=(40,10), sticky="ew")
        # self.validate = ctk.CTkButton(
        #         self,
        #         text="Save",
        #         fg_color="#D9DFC6",
        #         hover=False,
        #         text_color="#424242",
        #         font=("Inter", 20, "bold"),
        #         height=60,
        #         command=self.validate_effect

        #     )
        # self.validate.grid(row=2, column=1,pady=10, padx=(10,40))

        # self.compile_button = ctk.CTkButton(
        #     self,
        #     text="Save",
        #     fg_color="#D9DFC6",
        #     hover=False,
        #     text_color="#424242",
        #     font=("Inter", 20, "bold"),
        #     height=60,
        #     command=self.compile_to_file
        # )
        # self.compile_button.grid(row=2, column=1,pady=10, padx=(10,40))
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=2, column=1, padx=(10, 40), pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1), weight=1)  # two equal columns

        # ✅ Validate Button
        self.validate_button = ctk.CTkButton(
            self.controls_frame,
            text="Validate",
            fg_color="#EFF3EA",
            hover=False,
            text_color="#424242",
            font=("Inter", 20, "bold"),
            height=60,
            command=self.validate
        )
        self.validate_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        # ✅ Compile Button
        self.compile_button = ctk.CTkButton(
            self.controls_frame,
            text="Save",
            fg_color="#D9DFC6",
            hover=False,
            text_color="#424242",
            font=("Inter", 20, "bold"),
            height=60,
            command=self.compile_to_file
        )
        self.compile_button.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        self.user_input.bind("<Return>", lambda event: self.generate_output())

        

    def add_chat_bubble(self, text, sender="user"):
        """Adds a chat bubble with pastel colors and modern font."""
        bubble_color = "#FFF2C2" if sender == "user" else "#FFD3B6"  
        text_align = "e" if sender == "user" else "w" 

        bubble = ctk.CTkLabel(self.chat_frame, text=text, wraplength=1000, justify="left",
                              fg_color=bubble_color, text_color="#424242", font=("Inter", 20, "normal"),
                              corner_radius=20, padx=25, pady=15)
        bubble.pack(anchor=text_align, pady=10, padx=40)


        # self.chat_frame._parent_canvas.yview_moveto(1)
        self.after(100, lambda: self.chat_frame._parent_canvas.yview_moveto(1))

    def generate_output(self):
        """Handles user input and displays chat bubbles."""
    
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        

        if self.welcome_message:
            self.welcome_message.destroy()
            self.welcome_message = None

        self.add_chat_bubble(f"{user_text}", sender="user")


        typing_bubble = ctk.CTkLabel(self.chat_frame, text="Generating...", wraplength=900, justify="left",
                                     fg_color="#FFD3B6", text_color="#333", font=("Inter", 20, "italic"),
                                     corner_radius=10, padx=20, pady=10)
        typing_bubble.pack(anchor="w", pady=10, padx=40)
        self.update_idletasks()

        def fetch_output():
  
            # for _ in range(3): 
            #     time.sleep(0.5)
            #     typing_bubble.configure(text="Generating" + "." * (_ % 3 + 1))
            #     self.update_idletasks()

     
            # if self.is_generating_effect:
            #         stream = self.effect_agent.generate_effect_stream(user_text)
            # else:
            #         stream = self.policy_agent.generate_policy_stream(user_text)

            if self.is_generating_effect:
            
                if not self.generated_effect_history:
                    stream = self.effect_agent.generate_effect_stream(user_text)
                else:
                    last_effect = self.generated_effect_history[-1]
                    stream = self.effect_agent.refine_effect_stream(last_effect, user_text)
            else:
                if not self.generated_policy_history:
                    stream = self.policy_agent.generate_policy_stream(user_text)
                else:
                    last_effect = self.generated_policy_history[-1]
                    stream = self.policy_agent.refine_policy_stream(last_effect, user_text)
              
      
            response_text = ""
            for chunk in stream:
                delta = chunk['message']['content']
                delta = chunk['message']['content'].replace("```", "")
                response_text += delta
                typing_bubble.configure(text=response_text)
                self.update_idletasks()

            if self.is_generating_effect:
                self.effect_text = response_text
                self.generated_effect_history.append(response_text)
            else:
                self.policy_text = response_text

                self.generated_policy_history.append(response_text)

            typing_bubble.destroy()
            self.add_chat_bubble(response_text, sender="bot")

        threading.Thread(target=fetch_output, daemon=True).start()
        self.user_input.delete(0, "end")

    def toggle_mode(self):
        """Toggles between Effect and Policy generation."""
        self.is_generating_effect = not self.is_generating_effect
        mode_text = "Effect" if self.is_generating_effect else "Policy"
        if self.is_generating_effect:
       
            new_color = "#FFD3B6"
            mode="Switch To Policy Mode"
            border_color="#FFD3B6"

           
        else:
           
            new_color = "#EFF3EA" 
            mode="Switch To Effect Mode"
            border_color="#D9DFC6"


          
        self.toggle_button.configure(
       
            fg_color=new_color,
            border_color=border_color, 
            
       
            text=mode
            
        )
        # self.user_input.configure(
        #     placeholder_text=f"Describe your {mode_text}..."
        # )
        self.user_input.delete(0, "end") 
        self.user_input.focus()
        self.user_input.update()
        self.user_input.icursor("end")
        self.user_input.select_clear()
        self.chat_container.configure(
            border_color=border_color,
        )


    def compile_to_file(self):
        """Compiles the generated effect and policy into a text file."""
        # if not self.effect_text or not self.policy_text:
        #     self.add_chat_bubble("Both Effect and Policy must be generated before compiling!", sender="bot")
        #     return

        file_content =f'import "{self.vocab}"'+f"\n{self.environment_constants}\n{self.effect_text}\n\n{self.policy_text}"
        file_path = f"./{self.env_name}/{self.env_name}_policy.rlang"

        with open(file_path, "w") as file:
            file.write(file_content)

        for widget in self.winfo_children():
            widget.destroy()


        # Configure root grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        preview_frame = ctk.CTkFrame(
            self,
            fg_color="#FFFDF0",
            corner_radius=20
        )
        preview_frame.grid(row=0, column=0, padx=40, pady=20, sticky="nsew")
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        # Load and insert file content into editable textbox
        with open(file_path, "r") as file:
            content = file.read()

        preview_textbox = ctk.CTkTextbox(
            preview_frame,
            wrap="word",
            font=("Inter", 18, "normal"),
            text_color="#424242",
            fg_color="white",
            corner_radius=15,
            border_width=2,
            border_color="#D9DFC6"
        )
        preview_textbox.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        preview_textbox.insert("1.0", content)
        preview_textbox.focus()


        def save_edited_file():
            updated_content = preview_textbox.get("1.0", "end-1c")
            with open(file_path, "w") as file:
                file.write(updated_content)
            preview_textbox.configure(border_color="#A2D5AB") 


        controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        controls_frame.grid(row=1, column=0, pady=(0, 30), sticky="ew")

        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_columnconfigure(1, weight=1)

        save_button = ctk.CTkButton(
            controls_frame,
            text="Save Changes",
            
            fg_color="#EFF3EA",
            text_color="#424242",
            font=("Inter", 20, "bold"),
            height=60,
            hover=False,
            corner_radius=10,
            command=save_edited_file
        )
        save_button.grid(row=0, column=0, padx=(55,10),sticky="ew")

        exit_button = ctk.CTkButton(
            controls_frame,
            text="Exit Preview",
      
            fg_color="#EFF3EA",
            text_color="#424242",
            font=("Inter", 20, "bold"),
            height=60,
            hover=False,
            corner_radius=10,
            command=self.destroy  # or reinitialize your app if needed
        )
        exit_button.grid(row=0, column=1, padx=(10,55),sticky="ew")

    def validate(self):
        is_effect = self.is_generating_effect
        target_type = "Effect" if is_effect else "Policy"
        source_text = self.effect_text if is_effect else self.policy_text

        if not source_text:
            self.add_chat_bubble(f"⚠️ You need to generate a {target_type} before validating it.", sender="bot")
            return


        typing_bubble = ctk.CTkLabel(
            self.chat_frame,
            text="Validating.",
            wraplength=900,
            justify="left",
            fg_color="#FFD3B6",
            text_color="#333",
            font=("Inter", 20, "italic"),
            corner_radius=10,
            padx=20,
            pady=10
        )
        typing_bubble.pack(anchor="w", pady=10, padx=40)
        self.update_idletasks()

        loading_texts = ["Validating.", "Validating..", "Validating..."]
        animating = True

        def animate(i=0):
            if not animating:
                return
            typing_bubble.configure(text=loading_texts[i % len(loading_texts)])
            self.after(500, animate, i + 1)

        animate()

        def run_validation():
            nonlocal animating
            try:
                filename = f"./{self.env_name}/effect.rlang" if is_effect else f"./{self.env_name}/policy.rlang"
                with open(filename, "w") as f:
                    if not is_effect:
                        f.write(f'import "{self.vocab}"\n'+self.environment_constants.strip() + "\n\n" + source_text.strip())
                    else:
                        f.write(self.environment_constants.strip() + "\n\n" + source_text.strip())

                venv_python_path = "../../.venv/bin/python"
                validation_script_path = f"../validator.py"

                os.chdir(self.env_name)
            
                print(f"[DEBUG] Current working directory: {os.getcwd()}")

                
                args = [venv_python_path, validation_script_path, filename,self.env_name,"1" if is_effect else "0"]
         
                if not is_effect:
                    policy_name = extract_policy_name(source_text)
                    args.append(policy_name)
                else:
                    args.append("0")

                result = subprocess.run(
                    args=args,
                    capture_output=True,
                    text=True
                )

                output = result.stdout.strip()
                error_output = result.stderr.strip()
                print(output)
                print(error_output)
                os.chdir("..")


                animating = False
                self.after(0, typing_bubble.destroy)
                if "VALID" in output:
                    self.add_chat_bubble(f"{target_type} passed validation!", sender="bot")
                elif "ERROR" in output:
                    print(output)
                    self.add_chat_bubble(f"{target_type} validation failed:\n Refine Your {target_type}", sender="bot")
                else:
                    self.add_chat_bubble(f"⚠️ Unexpected validator output:\n{output or error_output}", sender="bot")

            except Exception as e:
                        animating = False
                        self.after(0, typing_bubble.destroy)
                        self.add_chat_bubble(f"Validation subprocess error:\n{e}", sender="bot")

        threading.Thread(target=run_validation, daemon=True).start()
from PIL import Image
class SelectEnvironmentScreen(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Choose Your Environment")
        self.attributes("-fullscreen", True) 
        self.configure(fg_color="#FFFDF0")
        ctk.set_appearance_mode("light")
        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))  # Allow exiting fullscreen

        title = ctk.CTkLabel(self, text="Choose an Environment", font=("Inter", 28, "bold"), text_color="#424242")
        title.pack(pady=(30, 10))

        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(expand=True, fill="both", padx=50, pady=20)

        # Grid setup
        container.grid_columnconfigure((0, 1), weight=1)

        # Taxi
        self._create_env_card(
            parent=container,
            column=0,
            image_path="assets/taxi.png",
            name="Taxi",
            callback=lambda: self.launch_env("taxi")
        )

        # CliffWalking
        self._create_env_card(
            parent=container,
            column=1,
            image_path="assets/cliff.png",
            name="Cliff Walking",
            callback=lambda: self.launch_env("cliff_walking")
        )

    def _create_env_card(self, parent, column, image_path, name, callback):
        frame = ctk.CTkFrame(parent, corner_radius=15, fg_color="#F4F4F4")
        frame.grid(row=0, column=column, padx=20, pady=20, sticky="nsew")

        # image = ctk.CTkImage(light_image=image_path, size=(220, 220))
        img = Image.open(image_path)
        image = ctk.CTkImage(light_image=img, size=(220, 220))
        img_label = ctk.CTkLabel(frame, image=image, text="")
        img_label.image = image
        img_label.pack(pady=(10, 5))

        label = ctk.CTkLabel(frame, text=name, font=("Inter", 22, "bold"), text_color="#333")
        label.pack(pady=(5, 10))

        button = ctk.CTkButton(
            frame,
            text=f"Use {name}",
            fg_color="#FFD3B6",
            text_color="#424242",
            font=("Inter", 18, "bold"),
            command=callback
        )
        button.pack(pady=(0, 15))

    def launch_env(self, env_name):
        self.destroy()
        vocab_path = "vocab.json"


        if env_name == "taxi":
            env_defs = constants.environment_definitions_taxi
            effect_agent = EffectAgent(constants.effect_prompt, constants.taxi_effect_fewshots, env_defs)
            policy_agent = PolicyAgent(constants.policy_prompt, constants.taxi_policy_fewshots, env_defs, vocab=vocab_path)
        else:  # cliff
            env_defs = constants.environment_definitions_cliff_walking
            effect_agent = EffectAgent(constants.effect_prompt, constants.cliff_walking_effect_fewshots, env_defs)
            policy_agent = PolicyAgent(constants.policy_prompt, constants.cliff_walking_policy_fewshots, env_defs,vocab=vocab_path)

        effect_agent.start_ollama_serve()

        app = ChatApp(effect_agent, policy_agent, environment_constants=env_defs, vocab=vocab_path,env_name=env_name)
        app.mainloop()
        effect_agent.stop_ollama_serve()


import customtkinter as ctk
import threading
import subprocess
import json

class TrainingChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Q-Learning Training Monitor")
        self.geometry("800x400")
        self.configure(fg_color="#FFFDF0")
        ctk.set_appearance_mode("light")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="🚖 Q-Learning Training Progress",
            font=("Inter", 24, "bold"),
            text_color="#424242"
        )
        self.title_label.pack(pady=(10, 20))

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.main_frame, width=600)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10)

        # Status
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Click 'Start Training' to begin...",
            font=("Inter", 18),
            text_color="#424242"
        )
        self.status_label.pack(pady=(10, 20))

        # Button
        self.train_button = ctk.CTkButton(
            self.main_frame,
            text="Start Training",
            fg_color="#FFD3B6",
            text_color="#424242",
            font=("Inter", 20, "bold"),
            height=50,
            command=self.start_training
        )
        self.train_button.pack(pady=10)

    def start_training(self):
        self.train_button.configure(state="disabled")
        self.status_label.configure(text="Initializing training...")

        def run_training():
            params = {
                "episodes": 15000,
                "use_knowledge": True,
                "rlang_path": "./taxi.rlang",
                "plot_path": "./plots/q_learning_with_knowledge.png"
            }

            process = subprocess.Popen(
                ["/Library/Frameworks/Python.framework/Versions/3.7/bin/python3", "run_agent.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="<path_to_script_directory>",
                bufsize=1,
                universal_newlines=True
            )

            process.stdin.write(json.dumps(params))
            process.stdin.flush()
            process.stdin.close()

            for line in process.stdout:
                try:
                    update = json.loads(line.strip())
                    if update.get("done"):
                        self.status_label.configure(
                            text=f"✅ Training complete!\nAverage Test Reward: {update['final_avg_reward']:.2f}"
                        )
                        self.progress_bar.set(1.0)
                        break

                    ep = update["episode"]
                    avg_r = update["avg_reward"]
                    progress = ep / params["episodes"]
                    self.progress_bar.set(progress)
                    self.status_label.configure(
                        text=f"Episode {ep}/{params['episodes']} | Avg Reward (last 100): {avg_r:.2f}"
                    )
                    self.update_idletasks()

                except Exception as e:
                    print("Error:", e, line)

            process.stdout.close()
            process.wait()

        threading.Thread(target=run_training, daemon=True).start()



      

# if __name__ == "__main__":
    

#     stage1 = EffectAgent(system_prompt=constants.effect_prompt, few_shots=constants.effect_fewshots, environment_definitions=constants.environment_definitions)
#     print(stage1)
#     stage2 = PolicyAgent(system_prompt=constants.policy_prompt, few_shots=constants.policy_fewshots, environment_definitions=constants.environment_definitions_policy,vocab="./vocab.json")

#     try:
#         stage1.start_ollama_serve()
#         app = ChatApp(stage1, stage2,environment_constants=constants.environment_definitions,vocab="vocab.json")
#         app.mainloop()
#     finally:
#         stage1.stop_ollama_serve()
#     # if __name__ == "__main__":
#     # app = TrainingChatApp()
#     # app.mainloop()

if __name__ == "__main__":
    app = SelectEnvironmentScreen()
    app.mainloop()