from tkinter import Tk, Button
from game_gui import GameGrid
from expectimax import ExpectiMax

if __name__ == '__main__':
    window = Tk()
    agent = ExpectiMax()

    game_grid = GameGrid(window, agent)
    game_grid.pack()

    window.mainloop()