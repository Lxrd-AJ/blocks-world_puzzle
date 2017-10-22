from problem import *

_AGENT_ = "AGENT"
#Define the initial problem state
initial_state = list(range(16))
initial_state[15] = _AGENT_
initial_state[14] = 'C'
initial_state[13] = 'B'
initial_state[12] = 'A'
print(initial_state)
#Define the goal state
goal_state = list(range(16))
goal_state[5] = 'A'
goal_state[9] = 'B'
goal_state[13] = 'C'
print(goal_state)

blocksword = BlocksWorldProblem(initial_state, goal_state)
print(blocksword)