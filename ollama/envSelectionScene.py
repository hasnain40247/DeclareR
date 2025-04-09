import customtkinter as ctk
from PIL import Image
from llm_agents.EffectAgent import EffectAgent
from llm_agents.PolicyAgent import PolicyAgent
import constants
from rlangChatScene import RLangChatScene

class EnvSelectionScene(ctk.CTk):
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

        container.grid_columnconfigure(0, weight=1)  
        container.grid_columnconfigure(1, weight=1) 
        container.grid_columnconfigure(2, weight=1)  


        # Taxi
        self._create_env_card(
            parent=container,
            column=0,
            image_path="assets/taxi.png",
            name="Taxi",
            callback=lambda: self.launch_env("taxi"),
            row=0
        )

        self._create_env_card(
            parent=container,
            column=2,
            image_path="assets/cliff.png",
            name="Cliff Walking",
            callback=lambda: self.launch_env("cliff_walking"),
            row=0
        )
        self._create_env_card(
    parent=container,
    column=1,
    image_path="assets/frozen_lake.png",
    name="Frozen Lake",
    callback=lambda: self.launch_env("frozen_lake"),
    row=1
)


    def _create_env_card(self, parent, column, image_path, name, callback, row=0, columnspan=1):
        frame = ctk.CTkFrame(parent, corner_radius=15, fg_color="#F4F4F4")
        frame.grid(row=row, column=column, columnspan=columnspan, padx=20, pady=20, sticky="nsew")

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
        elif env_name == "frozen_lake":
            env_defs = constants.environment_definitions_frozen_lake
            effect_agent = EffectAgent(constants.effect_prompt, constants.frozen_lake_effect_fewshots, env_defs)
            policy_agent = PolicyAgent(constants.policy_prompt, constants.frozen_lake_policy_fewshots, env_defs,vocab=vocab_path)

        
        else:  # cliff
            env_defs = constants.environment_definitions_cliff_walking
            effect_agent = EffectAgent(constants.effect_prompt, constants.cliff_walking_effect_fewshots, env_defs)
            policy_agent = PolicyAgent(constants.policy_prompt, constants.cliff_walking_policy_fewshots, env_defs,vocab=vocab_path)

        effect_agent.start_ollama_serve()

        app = RLangChatScene(effect_agent, policy_agent, environment_constants=env_defs, vocab=vocab_path,env_name=env_name)
        app.mainloop()
        effect_agent.stop_ollama_serve()



if __name__ == "__main__":
    app = EnvSelectionScene()
    app.mainloop()

