environment_definitions="""
Constant passenger_locs := [[0, 0], [0, 4], [4, 0], [4, 3]]
Constant destination_locs := [[0, 0], [0, 4], [4, 0], [4, 3]]

Factor taxi_position := S[0, 1]  
Factor x := taxi_position[0]
Factor y := taxi_position[1]

Factor passenger_location := S[2]  
Factor destination := S[3] 

Proposition at_passenger := taxi_position in passenger_locs
Proposition at_destination := taxi_position in destination_locs
Proposition carrying_passenger := passenger_location == 4

Action move_s := 0
Action move_n := 1
Action move_e := 2
Action move_w := 3
Action pick_up := 4
Action drop_off := 5
        """


effect_prompt="""Your task is to translate natural language advice to RLang effect, which is a prediction about the
state of the world or the reward function. For each instance, we provide a piece of advice in natural language,
a list of allowed primitives, and you should complete the instance by filling the missing effect function.
Don’t use any primitive outside the provided primitive list."""

policy_prompt="""
Your task is to translate natural language advice to RLang policy, which is a direct function
from states to actions. For each instance, we provide a piece of advice in natural language, a
list of allowed primitives, and you should complete the instance by filling the missing policy
function. Don’t use any primitive outside the provided primitive list.
"""
effect_fewshots=""""
Advice = "Picking up the passenger updates their location."
Primitives = [at_passenger, pick_up, carrying_passenger,passenger_location]
Effect=
Effect main:
    if at_passenger and A == pick_up:
        S'.passenger_location -> 4

Advice = "Dropping off the passenger at the wrong location incurs a penalty."
Primitives = [carrying_passenger, at_destination, drop_off]
Effect=
Effect main:
    if carrying_passenger and A == drop_off:
        if at_destination:
            Reward 20
        else:
            Reward -10

Advice = "Attempting pick-up or drop-off at the wrong place leads to a penalty."
Primitives = [pick_up, drop_off]
Effect=
Effect main:
    if A == pick_up or A == drop_off:
        Reward -10

Advice = "Prevent movement beyond grid boundaries."
Primitives = [x, y, move_n, move_s, move_e, move_w]
Effect=
Effect main:
    if y == 0 and A == move_n:  
        S' -> S
    if y == 4 and A == move_s:  
        S' -> S
    if x == 0 and A == move_w:  
        S' -> S
    if x == 4 and A == move_e:  
        S' -> S

Advice = "The taxi should only pick up the passenger at the correct location and drop them off at the correct destination. Incorrect pick-ups or drop-offs incur a penalty. Movement has a small cost, and the taxi cannot move beyond the grid boundaries."
Primitives = [at_passenger, pick_up, carrying_passenger, at_destination, drop_off, x, y, move_n, move_s, move_e, move_w]
Effect=
Effect main:
    if at_passenger and A == pick_up:
        S'.passenger_location -> 4
    if carrying_passenger and A == drop_off:
        if at_destination:
            Reward 20
        else:
            Reward -10
    elif A == pick_up or A == drop_off:
        Reward -10
    else:
        Reward -2  
        if y == 0 and A == move_n:  
            S' -> S
        if y == 4 and A == move_s:  
            S' -> S
        if x == 0 and A == move_w:  
            S' -> S
        if x == 4 and A == move_e:  
            S' -> S
"""


policy_fewshots="""

Advice = "If you are at the passenger's location, pick them up."
Primitives = [at_passenger, pick_up,carrying_passenger]
Policy =
Policy main:
    if at_passenger and not carrying_passenger:
        Execute pick_up

Advice = "If you are carrying the passenger and at the destination, drop them off."
Primitives = [carrying_passenger,at_destination,drop_off]
Policy =
Policy main:
    if at_destination and carrying_passenger:
        Execute drop_off

Advice = "If you are carrying the passenger but not at the destination, move toward the destination."
Primitives = ['move_n', 'move_s', 'move_e', 'move_w', carrying_passenger,at_destination,destination_x,destination_y,x,y]
Policy =
Policy main:
    if carrying_passenger and at_destination:
        if x < destination_x:
            Execute move_e
        elif x > destination_x:
            Execute move_w
        elif y < destination_y:
            Execute move_n
        elif y > destination_y:
            Execute move_s

Advice = "If you are not carrying the passenger, move toward their location."
Primitives = ['move_n', 'move_s', 'move_e', 'move_w',passenger_x,passenger_y,x,y,carrying_passenger]
Policy =
Policy main:
    if not carrying_passenger:
        if x < passenger_x:
            Execute move_e
        elif x > passenger_x:
            Execute move_w
        elif y < passenger_y:
            Execute move_n
        elif y > passenger_y:
            Execute move_s
"""