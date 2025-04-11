environment_definitions_taxi="""

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

environment_definitions_cliff_walking="""
Constant grid_height := 4
Constant grid_width := 12
Constant goal_loc := [11, 3]

Factor agent_pos := S[0, 1]
Factor x := agent_pos[0]
Factor y := agent_pos[1]

Proposition at_goal := agent_pos == goal_loc
Proposition is_on_cliff := (y == 3) and (x >= 1) and (x <= 10)


Action move_n := 0
Action move_e := 1
Action move_s := 2
Action move_w := 3
        """
environment_definitions_frozen_lake="""
Constant frozen_locs := [[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2], [3, 1], [3, 2]]
Constant hole_locs := [[1, 1], [1, 3], [2, 3], [3, 0]]

Factor position := S[0, 0]
Factor row := position[0]
Factor col := position[1]
Factor state := row * 4 + col

Proposition reached_goal := col == 3 && row == 3
Proposition in_hole := position in hole_locs

Action left := 0
Action down := 1
Action right := 2
Action up := 3
"""

effect_prompt="""Your task is to translate natural language advice to RLang effect, which is a prediction about the
state of the world or the reward function. For each instance, we provide a piece of advice in natural language,
a list of allowed primitives, and you should complete the instance by filling the missing effect function.
Don’t use any primitive outside the provided primitive list."""

taxi_effect_fewshots=""""
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

policy_prompt="""
Your task is to translate natural language advice to RLang policy, which is a direct function
from states to actions. For each instance, we provide a piece of advice in natural language with the name of the policy, a
list of allowed primitives, and you should complete the instance by filling the missing policy
function. 

# Rules
 - Don’t use any primitive outside the provided primitive list.
 - Name the policy with the name provided in the advice strictly. If no name is given, name it 'main'.
"""

taxi_policy_fewshots="""

Advice = "If you are at the passenger's location, pick them up. Name the policy passenger_pickup"
Primitives = [at_passenger, pick_up,carrying_passenger]
Policy =
Policy passenger_pickup:
    if at_passenger and not carrying_passenger:
        Execute pick_up

Advice = "If you are carrying the passenger and at the destination, drop them off."
Primitives = [carrying_passenger,at_destination,drop_off]
Policy =
Policy main:
    if at_destination and carrying_passenger:
        Execute drop_off

Advice = "If you are carrying the passenger but not at the destination, move toward the destination. Name this policy destination_policy"
Primitives = ['move_n', 'move_s', 'move_e', 'move_w', carrying_passenger,at_destination,destination_x,destination_y,x,y]
Policy =
Policy destination_policy:
    if carrying_passenger and at_destination:
        if x < destination_x:
            Execute move_e
        elif x > destination_x:
            Execute move_w
        elif y < destination_y:
            Execute move_n
        elif y > destination_y:
            Execute move_s

Advice = "If you are not carrying the passenger, move toward their location. I want this to be called carry"
Primitives = ['move_n', 'move_s', 'move_e', 'move_w',passenger_x,passenger_y,x,y,carrying_passenger]
Policy =
Policy carry:
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


cliff_walking_policy_fewshots = """

Advice = "If you're at the bottom-left corner, move north to avoid falling. Name it climb_safely."
Primitives = [x, y, move_n]
Policy =
Policy climb_safely:
    if x == 0 and y == 3:
        Execute move_n

Advice = "If you're not at the goal, move toward the goal. Call this toward_goal."
Primitives = [x, y, goal_x, goal_y, at_goal, move_n, move_s, move_e, move_w]
Policy =
Policy toward_goal:
    if not at_goal:
        if x < goal_x:
            Execute move_e
        elif x > goal_x:
            Execute move_w
        elif y < goal_y:
            Execute move_s
        elif y > goal_y:
            Execute move_n

Advice = "If you are on the cliff, try to go up. Call this escape_cliff."
Primitives = [x, y, is_on_cliff, move_n]
Policy =
Policy escape_cliff:
    if is_on_cliff:
        Execute move_n


"""


cliff_walking_effect_fewshots = """

Advice = "Prevent movement beyond grid boundaries."
Primitives = [x, y, move_n, move_s, move_e, move_w, grid_width, grid_height]
Effect =
Effect main:
    if y == 0 and A == move_n:
        S' -> S
    if y == grid_height - 1 and A == move_s:
        S' -> S
    if x == 0 and A == move_w:
        S' -> S
    if x == grid_width - 1 and A == move_e:
        S' -> S

Advice = "Reaching the goal gives no penalty or reward."
Primitives = [at_goal]
Effect =
Effect main:
    if at_goal:
        Reward 0

Advice = "Falling off the cliff results in a heavy penalty."
Primitives = [is_on_cliff]
Effect =
Effect main:
    if is_on_cliff:
        Reward -100

Advice = "Default movement has a small penalty"
Primitives = [x, y, move_n, move_s, move_e, move_w, at_goal, is_on_cliff]
Effect =
Effect main:
    if at_goal:
        Reward 0
    elif is_on_cliff:
        Reward -100
    else:
        Reward -1
"""

frozen_lake_policy_fewshots = """
Advice = "If the agent is already at the goal, do nothing."
Primitives = [reached_goal]
Policy =
Policy at_goal:
    if reached_goal:
        Do nothing

Advice = "If the agent is above the goal, move down. If below, move up."
Primitives = [row]
Policy =
Policy vertical_alignment:
    if row < 3:
        Execute down
    elif row > 3:
        Execute up

Advice = "If the agent is left of the goal, move right. If right of the goal, move left."
Primitives = [col]
Policy =
Policy horizontal_alignment:
    if col < 3:
        Execute right
    elif col > 3:
        Execute left

Advice = "Prioritize horizontal alignment first, then vertical."
Primitives = [row, col]
Policy =
Policy main:
    if col < 3:
        Execute right
    elif row < 3:
        Execute down
"""


frozen_lake_policy_fewshots = """
Advice = "If the agent is already at the goal, do nothing."
Primitives = [reached_goal]
Policy =
Policy at_goal:
    if reached_goal:
        Do nothing

Advice = "If the agent is above the goal, move down. If below, move up."
Primitives = [row]
Policy =
Policy vertical_alignment:
    if row < 3:
        Execute down
    elif row > 3:
        Execute up

Advice = "If the agent is left of the goal, move right. If right of the goal, move left."
Primitives = [col]
Policy =
Policy horizontal_alignment:
    if col < 3:
        Execute right
    elif col > 3:
        Execute left

Advice = "Prioritize horizontal alignment first, then vertical."
Primitives = [row, col]
Policy =
Policy main:
    if col < 3:
        Execute right
    elif row < 3:
        Execute down
"""
