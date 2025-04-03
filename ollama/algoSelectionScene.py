import customtkinter as ctk

class AlgoSelectionScene(ctk.CTkFrame):
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


        container.grid_columnconfigure(0, weight=1)

  
        self._create_algorithm_card(
            parent=container,
            column=0,
            name="Q-Learning",
            callback=lambda: self.parent.launch_algorithm("Q-Learning")
        )


        self._create_algorithm_card(
            parent=container,
            column=0,
            name="Dyna-Q",
            callback=lambda: self.parent.launch_algorithm("Dyna-Q")
        )


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
        

        frame.grid(row=parent.grid_size()[1], column=column, padx=20, pady=20, sticky="nsew")
