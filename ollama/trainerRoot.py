
import customtkinter as ctk
from algoSelectionScene import AlgoSelectionScene
from renderScene import RenderScene


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
        

        self.selection_screen = AlgoSelectionScene(self)
    
    def launch_algorithm(self, algorithm_name):

        for widget in self.winfo_children():
            widget.destroy()
    
        self.renderer = RenderScene(self,self.env_name, algorithm_name)



if __name__ == "__main__":
    app = MainApplication(env_name="frozen_lake")
    app.mainloop()


