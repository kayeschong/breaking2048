import numpy as np

class RandomPlay():
    
    def get_move(self, board):
        return np.random.choice(board.get_valid_moves())  