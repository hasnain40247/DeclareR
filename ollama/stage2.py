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
  
            Advice = "If the passenger is in the taxi, go to their destination."
            Primitives = ['Taxi', 'Passenger', 'Destination', 'Location', 'Move', 'Pickup', 'Dropoff', 
                        'left', 'right', 'up', 'down', 'at', 'in_taxi']

            Policy main :
                if in_taxi ( Passenger ) :
                    Execute go_to ( Destination )

            Advice = "If the passenger is at the same location as you, pick them up."
            Primitives = ['Taxi', 'Passenger', 'Destination', 'Location', 'Move', 'Pickup', 'Dropoff', 
                        'left', 'right', 'up', 'down', 'at', 'in_taxi']

            Policy main :
                if at ( Passenger ) :
                    Execute Pickup

            Advice = "If you are at the destination with the passenger, drop them off."
            Primitives = ['Taxi', 'Passenger', 'Destination', 'Location', 'Move', 'Pickup', 'Dropoff', 
                        'left', 'right', 'up', 'down', 'at', 'in_taxi']

            Policy main :
                if in_taxi ( Passenger ) and at ( Destination ) :
                    Execute Dropoff

            Advice = "If the passenger is not in the taxi, go to them."
            Primitives = ['Taxi', 'Passenger', 'Destination', 'Location', 'Move', 'Pickup', 'Dropoff', 
                        'left', 'right', 'up', 'down', 'at', 'in_taxi']

            Policy main :
                if not in_taxi ( Passenger ) :
                    Execute go_to ( Passenger )

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
