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

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 2 ** self.n_tiles, shape=(self.height, self.width), dtype=np.uint32)

        self.reset()

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
            reward = 100
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
        self.grid = np.zeros(shape=(4, 4), dtype=np.uint32)

        self.grid = self.generate_tile(self.grid)
        self.grid = self.generate_tile(self.grid)
        return self.grid.copy()

    def render(self, mode='human'):
        # Render the environment to the screen
        print(self.grid)

    def generate_tile(self, grid):
        next_grid = grid.copy()
        if self._num_empty_tiles(next_grid) > 0:
            empty_positions = np.argwhere(next_grid.flatten() == 0).flatten()
            position_choice = np.random.choice(empty_positions)
            position_2d = np.unravel_index(position_choice, next_grid.shape)

            value_choice = np.random.choice([2, 4], p=[0.8, 0.2])

            next_grid[position_2d] = value_choice
        return next_grid

    def is_movable(self, direction):
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
        return (grid == 0).sum()

    def _game_lost(self, grid):
        # When no more valid moves
        lost = all([not self.is_movable(direction) for direction in range(4)])
        return lost

    def __str__(self):
        return np.array2string(self.grid)


if __name__ == '__main__':
    from tkinter import Tk, Frame

    # Focus must be on tkinter gui to activate commands
    main = Tk()
    game = Env2048()
    print(game)
    verbose = True


    def leftKey(event):
        print()
        game.step(0)
        print(game)


    def upKey(event):
        print()
        game.step(1)
        print(game)


    def rightKey(event):
        print()
        game.step(2)
        print(game)


    def downKey(event):
        print()
        game.step(3)
        print(game)


    frame = Frame(main, width=200, height=200)
    main.bind('<Left>', leftKey)
    main.bind('<Right>', rightKey)
    main.bind('<Up>', upKey)
    main.bind('<Down>', downKey)
    main.mainloop()
