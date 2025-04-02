from rlang.grounding import Feature

# def _agent_x(state):
#     return state[0] 

# def _agent_y(state):
#     return state[1] 

def _goal_x(state):
    return 11

def _goal_y(state):
    return 3


# Register these functions as Features
# agent_x = Feature(_agent_x)
# agent_y = Feature(_agent_y)
goal_x = Feature(_goal_x)
goal_y = Feature(_goal_y)

