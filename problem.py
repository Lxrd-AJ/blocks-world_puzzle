#!/usr/bin/python3

class BlocksWorldProblem(object):
    def __init__(self, initial, goal):
        self.initial = initial
        self.goal = goal 
        self.N = 4 #Puzzle size if fixed

    def actions(self, state):
        raise NotImplementedError

    def result(self, state):
        raise NotImplementedError

    # def goal_test(self, state):
    #     if isinstance(self.goal, list):
    #         return is_in(state, self.goal)
    