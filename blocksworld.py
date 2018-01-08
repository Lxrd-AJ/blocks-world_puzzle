from problem import *
from queue import Queue, LifoQueue, PriorityQueue
import numpy as np 
import json

_AGENT_ = "AGENT"
_CUT_OFF_ = 'CUT_OFF'
MAX_DEPTH = 5000000

#Define the initial problem state
# initial_state = list(range(16))
# initial_state[15] = _AGENT_
# # Original state definition
# initial_state[14] = 'C'
# initial_state[13] = 'B'
# initial_state[12] = 'A'

# Making the problem easier 
# initial_state[1] = 'A'
# initial_state[9] = 'B'
# initial_state[13] = 'C'

problem_difficulty = []
for a in range(4):
    for b in range(4,8):
        for c in range(8,12):
            puzzle = list(range(16))
            puzzle[15] = _AGENT_
            puzzle[a] = 'A'
            puzzle[b] = 'B'
            puzzle[c] = 'C'
            problem_difficulty.append(puzzle)

for a in range(0,12,4):
    for b in range(1,13,4):
        for c in range(2,14,4):
            puzzle = list(range(16))
            puzzle[15] = _AGENT_
            puzzle[a] = 'A'
            puzzle[b] = 'B'
            puzzle[c] = 'C'
            problem_difficulty.append(puzzle)

# Define the goal state
goal_state = list(range(16))
goal_state[5] = 'A'
goal_state[9] = 'B'
goal_state[13] = 'C'


def tree_search( problem, frontier=Queue(), fileH=None, randomise=False):
    initial_node = Node(problem.initial)
    frontier.put_nowait( initial_node )
    if problem.goal_test( initial_node.state ):
        return initial_node
    # explored = set() #Optimisation to avoid loops in frontier
    node_count = 1
    while not frontier.empty():
        node = frontier.get_nowait()
        # explored.add( str(node.state) )
        # utilities.observe( node )
        print(str(node) + " at depth: " + str(node.depth) + ", Num expanded = " + str(node_count))
        if fileH:
            fileH.write(str(node) + "\n")
        if problem.goal_test( node.state ):
            return node
        if node_count > MAX_DEPTH:
            return None
        for node in node.expand( problem , random=randomise):
            # if str(node.state) not in explored:
            frontier.put_nowait(node)
            node_count += 1
    return None

def bread_first_search( problem, fileH=None):
    return tree_search(problem, Queue(), fileH, randomise=True)

def depth_first_search( problem, fileH=None):
    return tree_search(problem, LifoQueue(), fileH, randomise=True)

def depth_limited_search( problem, limit=100, fileH=None ):
    def recursive_dls(node, problem, limit):
        print(str(node) + " = " + str(node.depth))
        if fileH:
            fileH.write(str(node) + "\n")
        if node.depth > MAX_DEPTH:
            return None
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

def iterative_deepening_search( problem, fileH=None ):
    for depth in range(300):
        print("IDS at Depth = " + str(depth))
        result = depth_limited_search( problem, depth, fileH )
        if result != _CUT_OFF_:
            return result
        else:
            print("***" * 3 + " No solution found")
            return None 
           
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
        if node.depth > MAX_DEPTH:
            return None
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

# goal = a_star_search( blocksworld )
# print("===" * 10)
# print("A* Search Goal State found")
# print(str(goal))


node_difficulty = {}
algorithms = [('BFS',bread_first_search),('DFS',depth_first_search),('IDS',iterative_deepening_search),('A*',a_star_search)]
# algorithms = reversed(algorithms)
for (difficulty,problem) in enumerate(problem_difficulty,start=1):
    node_difficulty[difficulty] = {}
    for (name,algorithm) in algorithms:
        print("Difficulty {:} - Search algorithm {:}".format(difficulty,name))
        blocksworld = BlocksWorldProblem(problem, goal_state, _AGENT_)
        goal_node =  algorithm(blocksworld)
        if goal_node:
            node_difficulty[difficulty][name] = goal_node.depth
        else:
            node_difficulty[difficulty][name] = -1
        with open('problem_difficulty.json','w') as file:
            file.write(json.dumps(node_difficulty))



"""
https://github.com/aimacode/aima-python/blob/master/search.py
__TODO__:
[x] Depth-first search
[x] Bread-first search
[x] Iterative deepening 
[x] A* heuristic method

[x] Deep reinforcement learning (Q-Learning) 
    * http://www.samyzaf.com/ML/rl/qmaze.html
    * https://github.com/khpeek/Q-learning-Hanoi
    * https://sandipanweb.wordpress.com/2017/03/24/solving-4-puzzles-with-reinforcement-learning-q-learning-in-python/
    * https://deepmind.com/blog/deep-reinforcement-learning/
    * http://www.cs.cmu.edu/~rsalakhu/10703/Lecture_DQL.pdf
    * http://hunch.net/~beygel/deep_rl_tutorial.pdf
    * https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
# Utilities
[ ] Utility function to show a state which is then added to tree search so a live view is available
[ ] Graphviz visualisation of how a tree is traversed
"""