#!/usr/bin/python3

class BlocksWorldProblem(object):
    def __init__(self, initial, goal, agent_marker):
        self.initial = initial
        self.goal = goal 
        self.agent_marker = agent_marker
        # implement state mapping
        """
        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15
        """
        self.UP = -4
        self.BACK = -1
        self.FORWARD = 1 
        self.DOWN = 4

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
        possible_actions = []
        agent_loc = self.agent_location(state)
        possible_states = self.state_mapping[agent_loc]
        
        for action in [self.UP, self.DOWN, self.FORWARD, self.BACK]:
            if (agent_loc + action) in possible_states:
                possible_actions.append(action)

        return possible_actions

    def agent_location(self, state):
        for i in range(0, len(state)):
            if self.agent_marker == state[i]:
                return i  
        print("** The impossible happened, the agent is not in the maze")
        return None 

    """
    The resulting state from applying the action in the current state
    """
    def result(self, state, action):
        state = list(state) #prevent pass by reference
        agent_loc = self.agent_location(state)
        next_agent_loc = agent_loc + action 
        #Check if there is a block in the new agent location
        if type(state[next_agent_loc]) is str:
            block = state[next_agent_loc]
            state[agent_loc] = block
        else:
            state[agent_loc] = agent_loc
        state[next_agent_loc] = self.agent_marker

        return state 

    def goal_test(self, state):
        if (state[5] == self.goal[5]) and (state[9] == self.goal[9]) and (state[13] == self.goal[13]):
            return True 
        else:
            return False

    def path_cost(self, c, state1, action, state2):
        return c + 1
    
class Node(object):
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state 
        self.parent = parent 
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def child_node( self, problem, action):
        next = problem.result(self.state, action)
        cost = problem.path_cost(self.path_cost, self.state, action, next)
        return Node(next, self, action, cost)

    def expand(self, problem):
        return [self.child_node(problem, action) for action in problem.actions(self.state)]

    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def solution(self):
        return [node.action for node in self.path()[1:]]
    
