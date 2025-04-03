import customtkinter as ctk
import time
import threading
class ReasoningBotFrame(ctk.CTkFrame):
    def __init__(self, parent, reasoning_agent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        self.reasoning_agent = reasoning_agent  # Store the ReasoningAgent instance
        self.chat_frame = ctk.CTkScrollableFrame(self, width=300, height=400, fg_color="#FFFDF0", corner_radius=25)
        self.chat_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.input_container = ctk.CTkFrame(self, fg_color="#EFF3EA", corner_radius=25, border_width=2)
        self.input_container.pack(fill="x", padx=20, pady=10)

        self.user_input = ctk.CTkEntry(
            self.input_container,
            placeholder_text="Type...",
            fg_color="transparent",
            text_color="#424242",
            font=("Inter", 16, "normal"),
            border_width=0,
            width=420
        )
        self.user_input.pack(side="left", fill="x", padx=10, pady=10)

        self.send_button = ctk.CTkButton(
            self.input_container,
            text="↵",
            width=35,
            height=35,
            fg_color="#D9DFC6",
            text_color="#424242",
            font=("Inter", 20, "bold"),
            corner_radius=100,
            hover=False,
            command=self.generate_output
        )
        self.send_button.pack(side="right", padx=10)

    def generate_output(self):
    
        user_text = self.user_input.get().strip()
        if user_text:
            self.add_chat_bubble(user_text, sender="user")

            state = "Some state information here"
            action = "Some action taken by the agent"
            rgb_array = None  


            self.stream_reasoning(state, action, rgb_array)
            self.user_input.delete(0, "end")

    def stream_reasoning(self, state, action, rgb_array):

        typing_bubble = ctk.CTkLabel(self.chat_frame, text="Thinking...", wraplength=250, justify="left",
                                     fg_color="#FFD3B6", text_color="#333", font=("Inter", 16, "italic"),
                                     corner_radius=10, padx=20, pady=10)
        typing_bubble.pack(anchor="w", pady=10, padx=40)
        self.update_idletasks()

        full_response = ""
        def update_typing_effect():
            nonlocal full_response
            for chunk in self.reasoning_agent.generate_reasoning_stream(state, action, rgb_array):
                full_response += chunk

                typing_bubble.configure(text=full_response)
                self.update_idletasks()  
                time.sleep(0.05) 

            typing_bubble.destroy()
            self.add_chat_bubble(full_response, sender="bot")


        threading.Thread(target=update_typing_effect, daemon=True).start()

    def add_chat_bubble(self, text, sender="user"):
        """Add a chat bubble to the chat frame"""
        bubble_color = "#FFF2C2" if sender == "user" else "#FFD3B6"
        text_align = "e" if sender == "user" else "w"

        bubble = ctk.CTkLabel(self.chat_frame, text=text, wraplength=250, justify="left",
                              fg_color=bubble_color, text_color="#424242", font=("Inter", 16, "normal"),
                              corner_radius=20, padx=25, pady=15)
        bubble.pack(anchor=text_align, pady=10, padx=40)
