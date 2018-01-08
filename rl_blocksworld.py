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
AGENT_MARKER = 47
_FREE_ = 0
_BLOCK_ = 10
_FLAG_ = 20
_TARGET_ = 40


_START_ = "_START_"
_OUT_OF_BOUNDS_ = "_OUT_OF_BOUNDS_"
_NO_ACTION_ = "_NO_ACTION_"
_PROGRESS_ = "_PROGRESS_"
_TERMINATE_ = "_TERMINATE_"
_IN_PROGRESS_ = "_IN_PROGRESS_"
_COMPLETE_ = "_COMPLETE_"
_MOVED_BLOCK_ = "_MOVED_BLOCK_"

"""
0 is a free cell
1 is a blocked location
"""
class BlocksWorld(object):
    def __init__(self, initial, agent_location=(0,0)):
        self._puzzle = initial
        nrows, ncols = self._puzzle.shape
        self.goal = (nrows-1,ncols-1)
        self.agent_location = agent_location
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._puzzle[r,c] == _FREE_]
        self.reset()

    def reset(self):
        self._puzzle,self.agent_location = self.random_puzzle_agent_location()
        self.puzzle = np.copy(self._puzzle)
        self.state = (self.agent_location[0], self.agent_location[1], _START_)
        self.min_reward = -0.5 * self.puzzle.size #-0.5 * self.puzzle.size
        self.total_reward = 0
        self.visited = set()

    def random_puzzle_agent_location(self):
        #TODO: Let the agent always start at (0,0)
        stride = self._puzzle.shape[0]
        puzzle = np.arange(self._puzzle.size).reshape((stride,stride))
        nrows, ncols = self._puzzle.shape
        self.goal = (nrows-1,ncols-1)
        puzzle[:] = 0
        puzzle[self.goal] = _TARGET_
        blocks = int(self._puzzle.size * 0.2)
        agent_location = (0,0)
        # Code commented out as agent location is now always at (0,0)
        # while True:
        #     agent_location = tuple(np.random.choice(stride, (1,2))[0])
        #     if agent_location != (nrows-1,ncols-1):
        #         puzzle[agent_location] = AGENT_MARKER
        #         break
        while blocks > 0:
            block_location = tuple(np.random.choice(stride, (1,2))[0])
            if block_location != agent_location \
                and puzzle[block_location] == _FREE_ \
                and block_location != (self.goal[0]-1,self.goal[1]) \
                and block_location != (self.goal[0], self.goal[1]-1): 
                puzzle[block_location] = _BLOCK_
                blocks += -1
        return puzzle,agent_location

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
        
        if not valid_actions:
            self.state = (self.state[0], self.state[1], _OUT_OF_BOUNDS_)
        elif action in valid_actions:            
            nrow, ncol = move(action, row, col)
            self.puzzle[row,col] = _FREE_
            self.puzzle[nrow,ncol] = AGENT_MARKER
            self.state = (nrow, ncol, _PROGRESS_)
        else:
            self.state = (self.state[0], self.state[1], _NO_ACTION_)
    
    def get_reward(self):
        if self.puzzle_solved():
            return 1.0 
        elif self.state[2] == _OUT_OF_BOUNDS_:
            return self.min_reward - 1
        elif (self.state[0], self.state[1]) in self.visited:
            return -0.25
        elif self.state[2] == _NO_ACTION_:
            return -0.75
        elif self.state[2] == _PROGRESS_:
            return -0.04

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        env_state = self.observe()
        return env_state, reward, status 

    def observe(self):
        canvas = self.draw_env()
        return canvas.reshape((1,-1)) 

    def draw_env(self, canvas=None):
        if canvas is None:
            canvas = np.copy(self.puzzle)
        # for (row,col) in self.visited:
        #     canvas[row,col] = 5
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
        nrows, ncols = self.puzzle.shape 
        if self.state[0] == nrows -1 and self.state[1] == ncols - 1:
            return True
        else:
            return False

    def random_valid_action(self):
        row,col,mode = self.state 
        actions = self.valid_actions_in_state(row,col)
        if actions:
            return random.choice(actions)
        else:
            return None 
    
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

        if row > 0 and self.puzzle[row-1,col] == _BLOCK_:
            actions.remove(UP)
        if row < num_rows-1 and self.puzzle[row+1,col] == _BLOCK_:
            actions.remove(DOWN)
        if col > 0 and self.puzzle[row,col-1] == _BLOCK_:
            actions.remove(LEFT)
        if col < num_cols-1 and self.puzzle[row,col+1] == _BLOCK_:
            actions.remove(RIGHT)

        return actions


class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
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
            envstate, action, reward, next_envstate, game_status, loss = self.memory[j]
            inputs[i] = envstate            
            targets[i] = self.predict(envstate)
            Q_sa = np.max(self.predict(next_envstate))
            if game_status == _COMPLETE_:
                targets[i, action] = reward
            else:
                targets[i, action] = (reward + self.discount * Q_sa) #- loss 
        return inputs, targets