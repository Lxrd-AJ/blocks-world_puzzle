from problem import *
from queue import Queue, LifoQueue
import numpy as np 
import matplotlib.pyplot as plt

_AGENT_ = "AGENT"

def show_maze( state, size ):
    # TODO: Incomplete
    fig, ax = plt.subplots()
    s = state 
    s[state.index('A')] = 20
    s[state.index('B')] = 30
    s[state.index('C')] = 40
    if _AGENT_ in state:
        s[state.index(_AGENT_)] = 50
    maze = np.array(s).reshape((size,size))
    print(maze)
    
    # plt.matshow( maze )
    # plt.show()
    
    # ax.matshow(maze, cmap=plt.cm.Blues)

    # for i in range(size):
    #     for j in range(size):
    #         c = maze[j,i]
    #         ax.text(i, j, str(c), va='center', ha='center')


#Define the initial problem state
initial_state = list(range(16))
initial_state[15] = _AGENT_
# Original state definition
# initial_state[14] = 'C'
# initial_state[13] = 'B'
# initial_state[12] = 'A'

# Making the problem easier 
initial_state[1] = 'A'
initial_state[9] = 'B'
initial_state[13] = 'C'

print(initial_state)
#Define the goal state
goal_state = list(range(16))
goal_state[5] = 'A'
goal_state[9] = 'B'
goal_state[13] = 'C'

blocksworld = BlocksWorldProblem(initial_state, goal_state, _AGENT_)

# show_maze( initial_state, 4 )

def tree_search( problem, frontier=Queue(), fileH=None):
    initial_node = Node(problem.initial)
    frontier.put_nowait( initial_node )
    if problem.goal_test( initial_node.state ):
        return initial_node
    explored = set() #Optimisation to avoid loops in frontier
    while not frontier.empty():
        node = frontier.get_nowait()
        explored.add( str(node.state) )
        print(node)
        if fileH:
            fileH.write(str(node) + "\n")
        if problem.goal_test( node.state ):
            return node
        for node in node.expand( problem ):
            if str(node.state) not in explored:
                frontier.put_nowait(node)
    return None

def bread_first_search( problem, fileH):
    return tree_search(problem, Queue(),fileH)

def depth_first_search( problem, fileH):
    return tree_search(problem, LifoQueue(), fileH)

with open('log_bfs.txt', 'w') as f:
    goal = bread_first_search( blocksworld, f )
    print("===" * 10)
    print("BFS Goal State found")
    print(str(goal))

# Depth-First Search - LIFO Queue
with open('log_dfs.txt', 'w') as f:
    goal = depth_first_search( blocksworld, f )
    print("===" * 10)
    print("DFS Goal State found")
    print(str(goal))

"""
https://github.com/aimacode/aima-python/blob/master/search.py
__TODO__:
[x] Depth-first search
[x] Bread-first search
[ ] Iterative deepening 
[ ] A* heuristic method
# Utilities
[ ] Utility function to show a state
"""