import customtkinter as ctk
import threading
import os
import subprocess
import re
from trainerRoot import MainApplication
def extract_policy_name(policy_text):
    match = re.search(r"Policy\s+(\w+):", policy_text)
    return match.group(1) if match else "unknown_policy"


class RLangChatScene(ctk.CTk):
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
        
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=2, column=1, padx=(10, 40), pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1), weight=1)  # two equal columns


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
  
            # if self.is_generating_effect:
            
            #     if not self.generated_effect_history:
            #         stream = self.effect_agent.generate_effect_stream(user_text)
            #     else:
            #         last_effect = self.generated_effect_history[-1]
            #         stream = self.effect_agent.refine_effect_stream(last_effect, user_text)
            # else:
            #     if not self.generated_policy_history:
            #         stream = self.policy_agent.generate_policy_stream(user_text)
            #     else:
            #         last_effect = self.generated_policy_history[-1]
            #         stream = self.policy_agent.refine_policy_stream(last_effect, user_text)
              
            agent = self.effect_agent if self.is_generating_effect else self.policy_agent

            stream = agent.generate_stream(user_text)  # Same method for both first + refined prompts
            response_text = ""

            for chunk in stream:
                delta = chunk["message"]["content"].replace("```", "")
                response_text += delta
                typing_bubble.configure(text=response_text)
                self.update_idletasks()

            agent.register_response(response_text)  # Add to chat memory
            # response_text = ""
            # for chunk in stream:
            #     delta = chunk['message']['content']
            #     delta = chunk['message']['content'].replace("```", "")
            #     response_text += delta
            #     typing_bubble.configure(text=response_text)
            #     self.update_idletasks()

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
        if not self.effect_text or not self.policy_text:
            self.add_chat_bubble("Both Effect and Policy must be generated before compiling!", sender="bot")
            return

        file_content =f'import "{self.vocab}"'+f"\n{self.environment_constants}\n{self.effect_text}\n\n{self.policy_text}"
        file_path = f"./{self.env_name}/{self.env_name}_policy_generated.rlang"

        with open(file_path, "w") as file:
            file.write(file_content)

        for widget in self.winfo_children():
            widget.destroy()



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
            self.destroy()  # Close the current window

            app = MainApplication(env_name=self.env_name)
            app.mainloop()  


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
            command=self.destroy 
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
                    self.add_chat_bubble(f"{target_type} validation failed:\n {error_output} \n Refine Your {target_type}", sender="bot")
                else:
                    self.add_chat_bubble(f"⚠️ Unexpected validator output:\n{output or error_output}", sender="bot")

            except Exception as e:
                        animating = False
                        self.after(0, typing_bubble.destroy)
                        self.add_chat_bubble(f"Validation subprocess error:\n{e}", sender="bot")

        threading.Thread(target=run_validation, daemon=True).start()
