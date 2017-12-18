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
initial_state[14] = 'C'
initial_state[13] = 'B'
initial_state[12] = 'A'
print(initial_state)
#Define the goal state
goal_state = list(range(16))
goal_state[5] = 'A'
goal_state[9] = 'B'
goal_state[13] = 'C'

blocksword = BlocksWorldProblem(initial_state, goal_state, _AGENT_)

# show_maze( initial_state, 4 )

#Breadth-First Search - FIFO Queue
def bread_first_search( problem, frontier=Queue(), fileH=None):
    frontier.put_nowait( Node(problem.initial) )
    while not frontier.empty():
        node = frontier.get_nowait()
        print(node)
        if fileH:
            fileH.write(str(node) + "\n")
        if problem.goal_test( node.state ):
            return node
        for node in node.expand( problem ):
            frontier.put_nowait(node)
    return None

with open('log_bfs.txt', 'w') as f:
    goal = bread_first_search( blocksword, Queue(), f )
    print("===" * 10)
    print("Goal State found")
    print(str(goal))

# Depth-First Search - LIFO Queue


"""
https://github.com/aimacode/aima-python/blob/master/search.py
__TODO__:
[ ] Depth-first search
[ ] Bread-first search
[ ] Iterative deepening 
[ ] A* heuristic method
# Utilities
[ ] Utility function to show a state
"""