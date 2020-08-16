import math
import time
import numpy as np

LEFT, UP, RIGHT, DOWN = range(4)

class ExpectiMax():

    def get_move(self, board):
        best_move, _ = self.maximize(board)
        return best_move

    def eval_board(self, board, n_empty): 
        grid = board.grid
        utility = 0
        smoothness = 0

        big_total = np.sum(np.power(grid, 2))
        # calculate smoothness of adjacent rows and columns
        s_grid = np.sqrt(grid)
        smoothness -= np.sum(np.abs(s_grid[::,0] - s_grid[::,1]))
        smoothness -= np.sum(np.abs(s_grid[::,1] - s_grid[::,2]))
        smoothness -= np.sum(np.abs(s_grid[::,2] - s_grid[::,3]))
        smoothness -= np.sum(np.abs(s_grid[0,::] - s_grid[1,::]))
        smoothness -= np.sum(np.abs(s_grid[1,::] - s_grid[2,::]))
        smoothness -= np.sum(np.abs(s_grid[2,::] - s_grid[3,::]))
        
        # weights
        empty_weight = 100000
        smoothness_weight = 3

        empty_u = n_empty * empty_weight
        smooth_u = smoothness ** smoothness_weight
        big_t_u = big_total

        # compute utility score
        utility += big_total
        utility += empty_u
        utility += smooth_u

        return (utility, empty_u, smooth_u, big_t_u)

    def maximize(self, board, depth = 0):
        moves = board.get_valid_moves()
        moves_boards = []

        # get grid state for each possible move
        for m in moves:
            m_board = board.clone()
            m_board.step(m)
            moves_boards.append((m, m_board))

        max_utility = (float('-inf'),0,0,0)
        best_direction = None

        # calculate utility of each possible state
        for mb in moves_boards:
            utility = self.chance(mb[1], depth + 1)
            
            # update utility
            if utility[0] >= max_utility[0]:
                max_utility = utility
                best_direction = mb[0]

        return best_direction, max_utility

    def chance(self, board, depth = 0):
        empty_cells = board.get_empty_cells()
        n_empty = len(empty_cells)

        #if n_empty >= 7 and depth >= 5:
        #    return self.eval_board(board, n_empty)

        if n_empty >= 6 and depth >= 3:
            return self.eval_board(board, n_empty)

        if n_empty >= 0 and depth >= 5:
            return self.eval_board(board, n_empty)

        if n_empty == 0:
            _, utility = self.maximize(board, depth + 1)
            return utility

        possible_tiles = []

        prob_2 = (.9 * (1 / n_empty))
        prob_4 = (.1 * (1 / n_empty))
        
        # get possible generated tile and porobability
        for empty_cell in empty_cells:
            possible_tiles.append((empty_cell, 2, prob_2))
            possible_tiles.append((empty_cell, 4, prob_4))

        utility_total = [0, 0, 0, 0]

        # try out all possible outcomes
        for t in possible_tiles:
            t_board = board.clone()
            t_board.insert_tile(t[0], t[1])
            _, utility = self.maximize(t_board, depth + 1)

            # sum (utility of each possible outcome * tile value)
            for i in range(4):
                utility_total[i] += utility[i] * t[2]

        return tuple(utility_total)
