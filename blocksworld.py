from problem import *
from queue import Queue, LifoQueue, PriorityQueue
import numpy as np 

_AGENT_ = "AGENT"
_CUT_OFF_ = 'CUT_OFF'

#Define the initial problem state
initial_state = list(range(16))
initial_state[15] = _AGENT_
# Original state definition
initial_state[14] = 'C'
initial_state[13] = 'B'
initial_state[12] = 'A'

# Making the problem easier 
# initial_state[1] = 'A'
# initial_state[9] = 'B'
# initial_state[13] = 'C'

print(initial_state)
#Define the goal state
goal_state = list(range(16))
goal_state[5] = 'A'
goal_state[9] = 'B'
goal_state[13] = 'C'

blocksworld = BlocksWorldProblem(initial_state, goal_state, _AGENT_)

# show_maze( initial_state, 4 )

"""
# TODO:
- [ ] Log the path cost for each node to file
- [ ] Might have to remove the `explored` set as it becomes graph search
"""
def tree_search( problem, frontier=Queue(), fileH=None):
    initial_node = Node(problem.initial)
    frontier.put_nowait( initial_node )
    if problem.goal_test( initial_node.state ):
        return initial_node
    explored = set() #Optimisation to avoid loops in frontier
    while not frontier.empty():
        node = frontier.get_nowait()
        explored.add( str(node.state) )
        print(str(node) + " = " + str(node.path_cost))
        if fileH:
            fileH.write(str(node) + "\n")
        if problem.goal_test( node.state ):
            return node
        for node in node.expand( problem ):
            if str(node.state) not in explored:
                frontier.put_nowait(node)
    return None

def bread_first_search( problem, fileH):
    return tree_search(problem, Queue(), fileH)

def depth_first_search( problem, fileH):
    return tree_search(problem, LifoQueue(), fileH)

def depth_limited_search( problem, limit=100, fileH=None ):
    def recursive_dls(node, problem, limit):
        print(str(node) + " = " + str(node.path_cost))
        if fileH:
            fileH.write(str(node) + "\n")
        if problem.goal_test( node.state ):
            return node 
        elif limit == 0:
            return _CUT_OFF_
        else:
            cutoff = False
            for child_node in node.expand( problem ):
                result = recursive_dls( child_node, problem, limit-1 )
                if result == _CUT_OFF_:
                    cutoff = True
                elif result is not None:
                    return result 
            return _CUT_OFF_ if cutoff else None 
    return recursive_dls( Node(problem.initial), problem, limit )

def iterative_deepening_search( problem, fileH ):
    for depth in range(300):
        print("IDS at Depth = " + str(depth))
        result = depth_limited_search( problem, depth, fileH )
        if result != _CUT_OFF_:
            return result
        else:
            print("***" * 3 + " No solution found")
           
def a_star_search( problem, fileH=None ):
    initial_node = Node(problem.initial)
    goal_node = Node(problem.goal)
    if problem.goal_test(initial_node.state):
        return initial_node 
    frontier = PriorityQueue()
    explored = set()
    frontier.put_nowait( (initial_node.path_cost,initial_node) )

    while not frontier.empty():
        next = frontier.get_nowait()
        node = next[1]
        explored.add( str(node.state) ) 
        print(str(node))
        print("Priority = " + str(next[0]))
        if problem.goal_test( node.state ):
            return node 
        for child in node.expand( problem ):
            if str(child.state) not in explored:
                # cost = node.path_cost + problem.manhattan_cost( node.state, child.state )
                goal_heuristic = problem.manhattan_cost( child.state, goal_node.state )
                priority = child.path_cost + goal_heuristic 
                frontier.put_nowait( (priority,child) )
    return None


# with open('log_bfs.txt', 'w') as f:
#     goal = bread_first_search( blocksworld, f )
#     print("===" * 10)
#     if goal is not None:
#         print("BFS Goal State found")
#         print(str(goal))
#     else:
#         print("No Goal state found")

# # Depth-First Search - LIFO Queue
# with open('log_dfs.txt', 'w') as f:
#     goal = depth_first_search( blocksworld, f )
#     print("===" * 10)
#     print("DFS Goal State found")
#     print(str(goal))

# with open('log_ids.txt', 'w') as f:
#     goal = iterative_deepening_search( blocksworld, f )
#     print("===" * 10)
#     print("Iterative Deepening Search Goal State found")
#     print(str(goal))

goal = a_star_search( blocksworld )
print("===" * 10)
print("A* Search Goal State found")
print(str(goal))

"""
https://github.com/aimacode/aima-python/blob/master/search.py
__TODO__:
[x] Depth-first search
[x] Bread-first search
[x] Iterative deepening 
[x] A* heuristic method

[ ] Deep reinforcement learning (Q-Learning) http://www.samyzaf.com/ML/rl/qmaze.html
# Utilities
[ ] Utility function to show a state
[ ] Graphviz visualisation of how a tree is traversed
"""