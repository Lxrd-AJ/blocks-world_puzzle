import numpy as np
import random

visited_color = 0.9
agent_color = 0.6
block_color = 0.3

#Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3 

actions = {
    LEFT: 'LEFT',
    RIGHT: 'RIGHT',
    UP: 'UP',
    DOWN: 'DOWN'
}
num_actions = len(actions.keys())

#Markers
EMPTY_MARKER = 0
AGENT_MARKER = 47
A_MARKER = 10
B_MARKER = 20
C_MARKER = 30
BLOCK_MARKERS = [A_MARKER, B_MARKER, C_MARKER]

_START_ = "_START_"
_OUT_OF_BOUNDS_ = "_OUT_OF_BOUNDS_"
_NO_ACTION_ = "_NO_ACTION_"
_PROGRESS_ = "_PROGRESS_"
_TERMINATE_ = "_TERMINATE_"
_IN_PROGRESS_ = "_IN_PROGRESS_"
_COMPLETE_ = "_COMPLETE_"
_MOVED_BLOCK_ = "_MOVED_BLOCK_"


class BlocksWorld(object):
    def __init__(self, initial, goal, agent_location=(3,3),convolution=False):
        self._puzzle = initial
        self.goal = goal
        nrows, ncols = self._puzzle.shape 
        self.agent_location = agent_location
        self.convolution = convolution
        self.reset()

    def reset(self):
        #TODO: Do not randomise the puzzle, rever to initial on reset
        # self._puzzle, agent_location = self.random_puzzle_agent_location()
        # self.agent_location = agent_location

        self.puzzle = np.copy(self._puzzle)
        self.state = (self.agent_location[0], self.agent_location[1], _START_)
        self.min_reward = -0.6 * self.puzzle.size #-0.5 * self.puzzle.size
        self.total_reward = 0
        self.visited = set()

    def random_puzzle_agent_location(self):
        stride = self._puzzle.shape[0]
        puzzle = np.arange(self._puzzle.size).reshape((stride,stride))
        puzzle[:] = 0
        for marker in BLOCK_MARKERS:
            not_found = True
            while not_found:
                random_location = tuple(np.random.choice(stride, (1,2))[0])
                if puzzle[random_location] == 0:
                    puzzle[random_location] = marker
                    not_found = False
                    break
        location = None
        while True:
            random_location = tuple(np.random.choice(stride, (1,2))[0])
            if not (puzzle[random_location] in BLOCK_MARKERS):
                location = random_location
                puzzle[random_location] = AGENT_MARKER
                break
        
        return (puzzle, location)

    def update_state(self, action):
        def move(action,row,col):
            if action == LEFT:
                col -= 1
            elif action == RIGHT:
                col += 1
            elif action == UP:
                row -= 1
            elif action == DOWN:
                row += 1
            return (row,col)
                
        row, col, mode = self.state 
        self.visited.add( (row,col) )
        valid_actions = self.valid_actions_in_state(row,col)
        
        if not action in valid_actions:
            self.state = (self.state[0], self.state[1], _OUT_OF_BOUNDS_)
        elif action in valid_actions:
            mode = _PROGRESS_
            nrow, ncol = move(action, row, col)
            if self.puzzle[nrow, ncol] in BLOCK_MARKERS:
                block = self.puzzle[nrow, ncol]
                self.puzzle[row,col] = block
                self.puzzle[nrow,ncol] = AGENT_MARKER
                mode = _MOVED_BLOCK_
            else:
                self.puzzle[row,col] = EMPTY_MARKER
                self.puzzle[nrow,ncol] = AGENT_MARKER
            self.state = (nrow, ncol, mode)
        else:
            self.state = (self.state[0], self.state[1], _NO_ACTION_)
    
    def get_reward(self):
        if self.puzzle_solved():
            return 1.0 
        elif self.state[2] == _OUT_OF_BOUNDS_:
            return self.min_reward - 1
        elif self.state[2] == _NO_ACTION_:
            return -0.75
        elif self.state[2] == _PROGRESS_:
            return 0 #-0.3 #-0.1
        elif self.state[2] == _MOVED_BLOCK_:
            #Should the agent be rewarded for stacking blocks irrespective of the positions/order/complteness
            reward = 0 #-0.1 #-0.5 #0.0

            A_block = self.search_for_marker( A_MARKER, self.goal )
            B_block = self.search_for_marker( B_MARKER, self.goal )
            C_block = self.search_for_marker( C_MARKER, self.goal )

            if (A_block[0]+1,A_block[1]) == B_block:
                reward += 0.15
            if (B_block[0]+1,B_block[1]) == C_block:
                reward += 0.15 

            return reward


    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        env_state = self.observe()
        return env_state, reward, status 

    def observe(self):
        canvas = self.draw_env()
        if self.convolution:
            canvas = np.array(canvas)
            print(canvas.shape)
            return canvas
        else:
            return canvas.reshape((1,-1)) 

    def draw_env(self, canvas=None):
        if canvas is None:
            canvas = np.copy(self.puzzle)
        rows,cols = self.puzzle.shape
        # for row in range(rows):
        #     for col in range(cols):
        #         if canvas[row,col] in BLOCK_MARKERS:
        #             canvas[row,col] = block_color
        #             print("Block found = " + str((row,col)))
        #         elif self.puzzle[row,col] == AGENT_MARKER:
        #             print("Agent found = "+ str((row,col)))
        #             canvas[row,col] = agent_color
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return _TERMINATE_ 
        elif self.puzzle_solved():
            return _COMPLETE_
        else:
            return _IN_PROGRESS_
                
    def search_for_marker( self, marker, puzzle):
            loc = np.where( puzzle == marker )
            return (loc[0][0], loc[1][0])

    def puzzle_solved(self):
        A_block = self.search_for_marker( A_MARKER, self.goal )
        B_block = self.search_for_marker( B_MARKER, self.goal )
        C_block = self.search_for_marker( C_MARKER, self.goal )

        if self.puzzle[A_block] == self.goal[A_block] and \
           self.puzzle[B_block] == self.goal[B_block] and \
           self.puzzle[C_block] == self.goal[C_block]:
            return True
        else: 
            return False

    def random_valid_action(self):
        row,col,mode = self.state 
        actions = self.valid_actions_in_state(row,col)
        return random.choice(actions)
    
    def valid_actions_in_state(self, row, col):
        actions = [UP, DOWN, LEFT, RIGHT]
        num_rows, num_cols = self.puzzle.shape 
        if row == 0:
            actions.remove(UP)
        elif row == num_rows-1:
            actions.remove(DOWN)

        if col == 0:
            actions.remove(LEFT)
        elif col == num_cols -1:
            actions.remove(RIGHT)

        #No need to check for road blocks as there are no blocking tiles in this game yet
        return actions


class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.85):
        self.model = model 
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, batch_size=100):
        env_size = self.memory[0][0].shape[1]
        mem_size = len(self.memory)
        batch_size = min(mem_size, batch_size)
        inputs = np.zeros((batch_size, env_size))
        targets = np.zeros((batch_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), batch_size, replace=False)):
            envstate, action, reward, next_envstate, game_status = self.memory[j]
            inputs[i] = envstate            
            targets[i] = self.predict(envstate)
            Q_sa = np.max(self.predict(next_envstate))
            if game_status == _COMPLETE_:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets