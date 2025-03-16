import ollama
import subprocess
import time
import os
class Stage3:
    def __init__(self, model="llama3:8b"):

        self.model = model

    def prompter(self, advice,primitives):
        prompt = f"""
            Your task is to translate natural language advice into an RLang effect, which makes predictions
            about the state of the world or the reward function. For each instance, we provide a piece of
            advice in natural language, a list of allowed primitives, and you should complete the instance
            by filling in the missing effect function. Don’t use any primitive outside the provided primitive
            list corresponding to each instance.

            Advice = "If you try to drive into a wall, nothing will happen."
            Primitives = ['Taxi', 'Passenger', 'Destination', 'Wall', 'Move', 'Pickup', 'Dropoff', 
                        'left', 'right', 'up', 'down', 'at', 'in_taxi']

            Effect main :
                if at ( Wall ) and A in [ left , right , up , down ] :
                    Reward 0
                    S’ -> S

            Advice = "If you drop off the passenger at the wrong location, you will receive a penalty."
            Primitives = ['Taxi', 'Passenger', 'Destination', 'Wall', 'Move', 'Pickup', 'Dropoff', 
                        'left', 'right', 'up', 'down', 'at', 'in_taxi']

            Effect main :
                if in_taxi ( Passenger ) and at ( X ) and X != Destination and A == Dropoff :
                    Reward -10

            Advice = "If you are not carrying a passenger, you can't drop them off."
            Primitives = ['Taxi', 'Passenger', 'Destination', 'Wall', 'Move', 'Pickup', 'Dropoff', 
                        'left', 'right', 'up', 'down', 'at', 'in_taxi']

            Effect main :
                if not in_taxi ( Passenger ) and A == Dropoff :
                    Reward 0
                    S’ -> S

            Advice = "Picking up a passenger when they are at your location will be successful."
            Primitives = ['Taxi', 'Passenger', 'Destination', 'Wall', 'Move', 'Pickup', 'Dropoff', 
                        'left', 'right', 'up', 'down', 'at', 'in_taxi']

            Effect main :
                if at ( Passenger ) and A == Pickup :
                    S’ -> in_taxi ( Passenger )
                    Reward +10

            Now give me the effect for the following:
            Advice = {advice}
            Primitives= {primitives}
            """

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()


    
    def start_ollama_serve(self):

        print("[\u2713] Starting Ollama server.")
        process=subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)
        return process

    def stop_ollama_serve(self,process):
     
        print("[\u2713] Stopping Ollama server.")
        process.terminate()  
        try:
            process.wait(timeout=5)  
        except subprocess.TimeoutExpired:
            process.kill()  

if __name__ == "__main__":

    stage3 = Stage3()  
    ollama_process = stage3.start_ollama_serve()
    advice = "Walking into balls is pointless. You will die if you walk into keys. Trying to open a" \
     "box when you aren’t near it will do nothing."
    primitives = ["Agent", "Wall", "GoalTile", "Lava", "Key", "Door", "Box", "Ball", "left", "right",
            "forward", "pickup", "drop", "toggle", "done", "pointing_right", "pointing_down", "pointing_left",
            "pointing_up", "go_to", "step_towards", "green_ball", "green_box", "purple_box", "agent",
            "purple_ball", "at", "reachable", "carrying"]
            
   
    print(stage3.prompter(advice, primitives))

    stage3.stop_ollama_serve(ollama_process)
