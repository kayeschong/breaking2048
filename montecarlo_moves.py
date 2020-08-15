import numpy as np

class MonteCarloGreedyMoves():
    def __init__(self, use_epsilon):
        self.use_epsilon = use_epsilon
        self.epsilon_value = 0.1

    def get_move(self, board):
        move_value = np.zeros(4)

        for move in board.get_valid_moves():
            test_board = board.clone()
            test_board.step(move)
            for run in range(1,11):
                move_value[move] += (self.random_play_simulation_run(test_board) - move_value[move])/run

        if self.use_epsilon:
            # if using epsilon greedy
            if np.random.uniform() < self.epsilon_value:
                return self.get_random_valid_move(board)
            
        # pure greedy
        return np.argmax(move_value)


    def random_play_simulation_run(self, board):
        test_env = board.clone()
        done = False
        moves_lasted = 0
        
        while not done:
            try:
                action = self.get_random_valid_move(test_env)
                _, _, done, _ = test_env.step(action)
                moves_lasted += 1
            except:
                done = True
                
            if done:
                break
        return moves_lasted

    def get_random_valid_move(self, board):
        return np.random.choice(board.get_valid_moves())  