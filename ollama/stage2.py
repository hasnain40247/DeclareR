import ollama
import subprocess
import time
import os
class Stage2:
    def __init__(self, model="llama3:8b"):

        self.model = model

    def prompter(self, advice,primitives):
        prompt = f"""
            Your task is to translate natural language advice to RLang policy, which is a direct function
            from states to actions. For each instance, we provide a piece of advice in natural language, a
            list of allowed primitives, and you should complete the instance by filling the missing policy
            function. Don’t use any primitive outside the provided primitive list corresponding to each
            instance, e.g., if there is no ‘green_door’ in the primitive list you must not use “green_door’
            for the policy function.

            Advice = “If the yellow door is open, go through it and walk to the goal. Otherwise open the
            yellow door if you have the key."
            Primitives = [‘Agent’, ‘Wall’, ‘GoalTile’, ‘Lava’, ‘Key’, ‘Door’, ‘Box’, ‘Ball’, ‘left’, ‘right’,
            ‘forward’, ‘pickup’, ‘drop’, ‘toggle’, ‘done’, ‘pointing_right’, ‘pointing_down’, ‘pointing_left’,
            ‘pointing_up’, ‘go_to’, ‘step_towards’, ‘yellow_key’, ‘yellow_door’, ‘agent’, ‘goal’, ‘at’, ‘carrying’]
            
            Policy main :
                if yellow_door.is_open :
                    Execute go_to ( goal )
                elif carrying ( yellow_key ) and at ( yellow_door ) and not yellow_door . is_open :
                    Execute toggle
            
            Advice = “If you don’t have the key, go get it."
            Primitives = [‘Agent’, ‘Wall’, ‘GoalTile’, ‘Lava’, ‘Key’, ‘Door’, ‘Box’, ‘Ball’, ‘left’, ‘right’,
            ‘forward’, ‘pickup’, ‘drop’, ‘toggle’, ‘done’, ‘pointing_right’, ‘pointing_down’, ‘pointing_left’,
            ‘pointing_up’, ‘go_to’, ‘step_towards’, ‘grey_key’, ‘red_door’, ‘grey_door’, ‘agent’, ‘purple_ball’, ‘at’, ‘carrying’]
            
            Policy main :
                if at ( grey_key ):
                    Execute pickup
                elif not carrying ( grey_key ):
                    Execute go_to ( grey_key )
            
            Advice = “If you are carrying a ball and its corresponding box is closed, open the box if you
            are at it, otherwise go to the box if you can reach it."
            Primitives = [‘Agent’, ‘Wall’, ‘GoalTile’, ‘Lava’, ‘Key’, ‘Door’, ‘Box’, ‘Ball’, ‘left’, ‘right’,
            ‘forward’, ‘pickup’, ‘drop’, ‘toggle’, ‘done’, ‘pointing_right’, ‘pointing_down’, ‘pointing_left’,
            ‘pointing_up’, ‘go_to’, ‘step_towards’, ‘green_ball’, ‘green_box’, ‘purple_box’, ‘agent’,
            ‘purple_ball’, ‘at’, ‘reachable’, ‘carrying’]
            
            Policy main :
                if carrying ( green_ball ) and not green_box . is_open :
                    if at ( green_box ):
                        Execute toggle
                    elif reachable ( green_box ) :
                        Execute go_to ( green_box )

            Advice = “Drop any balls for boxes you can’t reach"
            Primitives = [‘Agent’, ‘Wall’, ‘GoalTile’, ‘Lava’, ‘Key’, ‘Door’, ‘Box’, ‘Ball’, ‘left’, ‘right’,
            ‘forward’, ‘pickup’, ‘drop’, ‘toggle’, ‘done’, ‘pointing_right’, ‘pointing_down’, ‘pointing_left’,
            ‘pointing_up’, ‘go_to’, ‘step_towards’, ‘green_ball’, ‘green_box’, ‘purple_box’, ‘agent’,
            ‘purple_ball’, ‘at’, ‘reachable’, ‘carrying’]

            Policy main :
                if carrying ( green_ball ) and not reachable ( green_box ) :
                    Execute drop
                if carrying ( purple_ball ) and not reachable ( purple_box ):
                    Execute drop
            
            Now give me the policy for the following:
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

    stage2 = Stage2()  
    ollama_process = stage2.start_ollama_serve()



    advice = "If the green box is closed, go to it and open it."
    primitives = [
    'Agent', 'Wall', 'GoalTile', 'Lava', 'Key', 'Door', 'Box', 'Ball', 'left', 'right',
    'forward', 'pickup', 'drop', 'toggle', 'done', 'pointing_right', 'pointing_down',
    'pointing_left', 'pointing_up', 'go_to', 'step_towards', 'green_box', 'agent',
    'at', 'reachable'
]

   
    print(stage2.prompter(advice, primitives))

    stage2.stop_ollama_serve(ollama_process)
