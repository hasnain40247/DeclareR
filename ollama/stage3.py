import ollama
import subprocess
import time
import os
class Stage3:
    def __init__(self, model="llama3:8b"):

        self.model = model

    def prompter(self, advice,primitives):
        prompt = f"""
            Your task is to translate natural language advice to RLang effect, which is a prediction about
            the state of the world or the reward function. For each instance, we provide a piece of advice
            in natural language, a list of allowed primitives, and you should complete the instance by
            filling the missing effect function. Don’t use any primitive outside the provided primitive list
            corresponding to each instance, e.g., if there is no ‘green_door’ in the primitive list you must
            not use ‘green_door’ for the effect function.

            Advice = “Don’t go to the door without the key"
            Primitives = [‘yellow_door’, ‘goal’, ‘pickup’, ‘yellow_key’, ‘toggle’, ‘go_to’, ‘carrying’, ‘at’]
            
            Effect main :
                if at ( yellow_door ) and not carrying ( yellow_key ):
                    Reward -1
            
            Advice = “Don’t walk into closed doors. If you’re tired, don’t go forward."
            Primitives = [‘Agent’, ‘Wall’, ‘GoalTile’, ‘Lava’, ‘Key’, ‘Door’, ‘Box’, ‘Ball’, ‘left’, ‘right’,
            ‘forward’, ‘pickup’, ‘drop’, ‘toggle’, ‘done’, ‘pointing_right’, ‘pointing_down’, ‘pointing_left’,
            ‘pointing_up’, ‘go_to’, ‘step_towards’, ‘green_ball’, ‘green_box’, ‘purple_box’, ‘agent’,
            ‘purple_ball’, ‘at’, ‘reachable’, ‘carrying’]
            
            Effect main :
                if at ( yellow_door ) and yellow_door . is_closed and A == forward :
                    Reward -1
                    S’ -> S
                elif tired () and A == forward :
                    Reward -1
            
            Advice = “Walking into balls is pointless. You will die if you walk into keys. Trying to open a
            box when you aren’t near it will do nothing."
            Primitives = [‘Agent’, ‘Wall’, ‘GoalTile’, ‘Lava’, ‘Key’, ‘Door’, ‘Box’, ‘Ball’, ‘left’, ‘right’,
            ‘forward’, ‘pickup’, ‘drop’, ‘toggle’, ‘done’, ‘pointing_right’, ‘pointing_down’, ‘pointing_left’,
            ‘pointing_up’, ‘go_to’, ‘step_towards’, ‘green_ball’, ‘green_box’, ‘purple_box’, ‘agent’,
            ‘purple_ball’, ‘at’, ‘reachable’, ‘carrying’]
            
            Effect main :
                if at ( Ball ) and A == forward :
                    Reward 0
                    S’ -> S
                elif at ( Key ) and A == forward :
                    Reward -1
                    S’ -> S*0
                elif at ( Box ) and A == toggle :
                    Reward 0
                    S’ -> S

                         
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
