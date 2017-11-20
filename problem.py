#!/usr/bin/python3

class BlocksWorldProblem(object):
    def __init__(self, initial, goal):
        self.initial = initial
        self.goal = goal 
        self.N = 4 #Puzzle size if fixed
        # implement state mapping
        """
        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15

        up is -4
        back is -1
        forward is +1
        down is +4
        """
        self.state_mapping = {
            0: [1,4],
            1: [0, 5, 2],
            2: [1, 6, 3],
            3: [2, 7],
            4: [0, 5, 8],
            5: [1, 4, 6, 9],
            6: [2, 5, 7, 10],
            7: [3, 6, 11],
            8: [4, 9, 12],
            9: [5, 8, 10, 13],
            10: [6, 9, 11, 14 ],
            11: [7, 10, 15],
            12: [8, 13],
            13: [12, 9, 14],
            14: [13, 10, 15],
            15: [14, 11]
        }

    def actions(self, state):
        raise NotImplementedError

    def result(self, state):
        raise NotImplementedError

    # def goal_test(self, state):
    #     if isinstance(self.goal, list):
    #         return is_in(state, self.goal)
    
class Node:
    def __init__(self, state, parent=Node, action=None, path_cost=0):
        self.state = state 
        self.parent = parent 
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def expand(self, problem):
        return [self.child_node(problem, action) for action in problem.actions(self.state)]
    
