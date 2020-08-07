import numpy as np


class Game2048:
    def __init__(self):
        self.grid = np.zeros(shape=(4, 4), dtype=int)

        self.grid = self.generate_tile(self.grid)
        self.grid = self.generate_tile(self.grid)

    def generate_tile(self, grid):
        next_grid = grid.copy()
        if self._num_empty_tiles(next_grid) > 0:
            empty_positions = np.argwhere(next_grid.flatten() == 0).flatten()
            position_choice = np.random.choice(empty_positions)
            position_2d = np.unravel_index(position_choice, next_grid.shape)

            value_choice = np.random.choice([2, 4], p=[0.8, 0.2])

            next_grid[position_2d] = value_choice
        return next_grid

    def move(self, direction, verbose=False):
        """
        0 = left
        1 = up
        2 = right
        3 = down
        """
        self.grid, moved = self._move(self.grid, direction)
        if verbose and self._game_lost(self.grid):
            print("Game Over!")
        return moved

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
    game = Game2048()
    print(game)
    verbose = True

    def leftKey(event):
        print()
        game.move(0, verbose)
        print(game)


    def upKey(event):
        print()
        game.move(1, verbose)
        print(game)


    def rightKey(event):
        print()
        game.move(2, verbose)
        print(game)


    def downKey(event):
        print()
        game.move(3, verbose)
        print(game)


    frame = Frame(main, width=200, height=200)
    main.bind('<Left>', leftKey)
    main.bind('<Right>', rightKey)
    main.bind('<Up>', upKey)
    main.bind('<Down>', downKey)
    main.mainloop()
