import numpy as np
import gym
from gym import spaces


class Env2048(gym.Env):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    def __init__(self, height=4, width=4):
        super().__init__()

        self.height = height
        self.width = width
        self.n_tiles = self.height * self.width

        #self.action_space = spaces.Discrete(4)
        self.action_space = np.array([0,1,2,3])
        self.action_map = {0:"left", 1:"up", 2:"right", 3:"down"}
        self.observation_space = spaces.Box(0, 2 ** self.n_tiles, shape=(self.height, self.width), dtype=np.int)
        self.reset()
    
    def clone(self):
        grid_copy = Env2048()
        grid_copy.grid = np.copy(self.grid)
        return grid_copy

    def step(self, action):
        # Execute one time step within the environment
        """
        0 = left
        1 = up
        2 = right
        3 = down
        """
        assert action in self.action_space, f"Invalid action {action} | {self.action_space}"
        self.grid, moved = self._move(self.grid, action)

        obs = self.grid.copy()
        if moved:
            reward = 1
        else:
            reward = -1
        done = self._game_lost(self.grid)
        info = {
            "has_moved": moved,
            "last_action": action,
        }
        return obs, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.grid = np.zeros(shape=(4, 4), dtype=np.int)

        self.grid = self.generate_tile(self.grid)
        self.grid = self.generate_tile(self.grid)
        return self.grid.copy()

    def render(self, mode='human'):
        # Render the environment to the screen
        print(self.grid)

    def get_highest_tile(self):
        return np.amax(self.grid)
    
    def get_valid_moves(self):
        '''
        Output:
            list[int]: Valid moves for current grid
        '''
        return np.array([move for move in range(4) if self.is_movable(move)])
    
    def get_invalid_moves(self):
        '''
        Output:
            list[int]: Invalid moves for current grid
        '''
        return np.array([move for move in range(4) if not self.is_movable(move)])
    
    def get_empty_cells(self):
        '''
        Output:
            list[tuples]: Positions of empty cells
        '''
        cells = []
        for x in range(4):
            for y in range(4):
                if self.grid[x][y] == 0:
                    cells.append((x,y))
        return cells

    def generate_tile(self, grid):
        '''
        Output:
            numpy array: New grid with randomly generated tile
        '''
        next_grid = np.copy(grid)
        if self._num_empty_tiles(next_grid) > 0:
            empty_positions = np.argwhere(next_grid.flatten() == 0).flatten()
            position_choice = np.random.choice(empty_positions)
            position_2d = np.unravel_index(position_choice, next_grid.shape)

            value_choice = np.random.choice([2, 4], p=[0.8, 0.2])

            next_grid[position_2d] = value_choice
        return next_grid

    def insert_tile(self, position, value):
        '''
        insert tile into position in grid
        '''
        self.grid[position[0]][position[1]] = value

    def is_movable(self, direction):
        '''
        Output:
            bool: If there is a valid move for current grid
        '''
        _, moved = self._move(self.grid, direction)
        return moved

    def _left(self, grid):
        next_grid = grid.copy()
        next_grid, flushed = self._flush_left(next_grid)
        next_grid, joined = self._join_left(next_grid)
        # Flush again if joined
        if joined:
            next_grid, _ = self._flush_left(next_grid)

        moved = flushed or joined
        if moved:
            next_grid = self.generate_tile(next_grid)
        return next_grid, moved

    def _move(self, grid, direction: int):
        next_grid = grid.copy()
        next_grid = np.rot90(next_grid, direction)
        next_grid, moved = self._left(next_grid)
        next_grid = np.rot90(next_grid, -direction)

        return next_grid, moved

    def _flush_left(self, grid):
        flushed_grid = np.zeros_like(grid)
        for i, row in enumerate(grid):
            non_zero = row[row != 0]
            n_non_zero = len(non_zero)
            flushed_grid[i, :n_non_zero] = non_zero

        flushed = (flushed_grid != grid).any()
        return flushed_grid, flushed

    def _join_left(self, grid):
        joined_grid = grid.copy()
        for col_index in range(joined_grid.shape[1] - 1):
            curr_col = joined_grid[:, col_index]
            next_col = joined_grid[:, col_index + 1]
            joinable = (curr_col == next_col) & (next_col != 0)
            curr_col[joinable] *= 2
            next_col[joinable] = 0

        joined = (grid != joined_grid).any()
        return joined_grid, joined

    def _num_empty_tiles(self, grid):
        '''
        Output:
            int: Number of empty tiles on group
        '''
        return (grid == 0).sum()

    def _game_lost(self, grid):
        '''
        Output:
            bool: If no more valid moves
        '''
        lost = all([not self.is_movable(direction) for direction in range(4)])
        return lost

    def __str__(self):
        return np.array2string(self.grid)
    
    def get_random_valid_move(self):
        return np.random.choice(self.get_valid_moves())  

    def random_play_simulation_run(self):
        test_env = self.clone()
        done = False
        no_moves = 0
        
        while not done:
            try:
                action = test_env.get_random_valid_move()
                observation, reward, done, info = test_env.step(action)
                no_moves += 1
            
            except:
                done = True
                
            if done:
                break
            
        return no_moves, test_env.get_highest_tile()
 
no_moves = []
scores = []

for run in range(100):
    env = Env2048()
    env.reset()
    last_move_no, score = env.random_play_simulation_run()
    no_moves.append(last_move_no)
    scores.append(score)
    env.close()

print("Average moves lasted by playing randomly: {}".format(np.mean(np.array(no_moves))))  
print("Avg score by playing randomly: {}".format(np.mean(np.array(scores))))


