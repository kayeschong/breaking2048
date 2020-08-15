from tkinter import Frame, Label, CENTER, OptionMenu, StringVar, Button
from random import randint
import time
import threading

from games import Env2048
from expectimax import ExpectiMax
# from controls_gui import Controls

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")
ACTION_MAP = {0:"left", 1:"up", 2:"right", 3:"down", 4:"None"}
AGENT_DICT = {'Expectimax': ExpectiMax()}

class GameGrid(Frame):
    def __init__(self, root):
        Frame.__init__(self, root)

        # self.grid()
        self.master.title('2048')
        self.grid_cells = []
        self.current_agent = StringVar(root)
        self.score = StringVar(root)
        self.direction = StringVar(root)
        self.move_count_label = StringVar(root)
        self.move_count = 0
        self.choices = {'Expectimax','RL','Monte Carlo'}

        self.init_controls(root)
        self.init_grid(root)
        self.board = Env2048()
        self.update_grid_cells()
        self.algo = AGENT_DICT['Expectimax']

        self.game_active = False
        self.run_game()
        self.mainloop()

    def run_game(self):
        while True:
            if not self.game_active:
                break
            else:
                move_direction = self.algo.get_move(self.board)
                self.board.step(move_direction)
                self.move_count += 1
                self.update_results(move_direction)
                self.update_grid_cells()

                if len(self.board.get_valid_moves()) == 0:
                    self.game_over_display()
                    break

                self.update()

    def reset_grid(self, algo):
        self.move_count = 0
        self.update_results(4)

        self.board = Env2048()
        self.update_grid_cells()
        self.algo = AGENT_DICT[algo]

        self.game_active = False
        self.button.config(text="Stop")
        self.onStartStop()
        
    def game_over_display(self):
        for i in range(4):
            for j in range(4):
                self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)

        self.grid_cells[1][1].configure(text="TOP",bg=BACKGROUND_COLOR_CELL_EMPTY)
        self.grid_cells[1][2].configure(text="4 TILES:",bg=BACKGROUND_COLOR_CELL_EMPTY)
        top_4 = list(map(int, reversed(sorted(list(self.board.grid.flatten())))))
        self.grid_cells[2][0].configure(text=str(top_4[0]), bg=BACKGROUND_COLOR_DICT[2048], fg=CELL_COLOR_DICT[2048])
        self.grid_cells[2][1].configure(text=str(top_4[1]), bg=BACKGROUND_COLOR_DICT[2048], fg=CELL_COLOR_DICT[2048])
        self.grid_cells[2][2].configure(text=str(top_4[2]), bg=BACKGROUND_COLOR_DICT[2048], fg=CELL_COLOR_DICT[2048])
        self.grid_cells[2][3].configure(text=str(top_4[3]), bg=BACKGROUND_COLOR_DICT[2048], fg=CELL_COLOR_DICT[2048])
        self.update()

    def init_controls(self, root):
        background = Frame(root, width=SIZE, height=SIZE)
        background.grid(column=2, row=1, sticky='NSEW', padx=50, pady=50)

        Label(background, text="Choose an agent").grid(row = 1)

        self.current_agent.set('Expectimax') # default option
        agent_menu = OptionMenu(background, self.current_agent, *self.choices)
        agent_menu.grid(row = 2)
        # link function to change dropdown
        self.current_agent.trace('w', self.change_dropdown)
         
        self.button = Button(background, text="Start", width=6, command=self.onStartStop)
        self.button.grid(row = 3)

        Label(background, text="Current score:").grid(row = 4, column=0, pady=(50,0), sticky='W')
        self.score.set(2048)
        Label(background, textvariable=self.score).grid(row = 4, column=1, pady=(50,0))

        Label(background, text="Move Count:").grid(row = 5, column=0, sticky='W')
        self.move_count_label.set(self.move_count)
        Label(background, textvariable=self.move_count_label).grid(row = 5, column=1,)

        Label(background, text="Move Direction:").grid(row = 6, column=0, sticky='W')
        self.direction.set("None")
        Label(background, textvariable=self.direction).grid(row = 6, column=1,)
    
    def init_grid(self, root):
        background = Frame(root, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid(column=1, row=1)

        for i in range(GRID_LEN):
            grid_row = []

            for j in range(GRID_LEN):

                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def change_dropdown(self, *args):
        self.reset_grid(self.current_agent.get())
        print( "dropdown:", self.current_agent.get() )

    def gen(self):
        return randint(0, GRID_LEN - 1)

    # def init_matrix(self):
    #     self.board = Env2048()
        # self.add_random_tile()
        # self.add_random_tile()

    def update_grid_cells(self):
        self.score.set(self.board.get_highest_tile())
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = int(self.board.grid[i][j])
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    n = new_number
                    if new_number > 2048:
                        c = 2048
                    else:
                        c = new_number

                    self.grid_cells[i][j].configure(text=str(n), bg=BACKGROUND_COLOR_DICT[c], fg=CELL_COLOR_DICT[c])
        self.update_idletasks()

    def update_results(self, move_direction):
        self.direction.set(ACTION_MAP[move_direction])
        self.move_count_label.set(self.move_count)
        self.board.render()
        print("Move direction:", ACTION_MAP[move_direction])
        print("Move count:", self.move_count)

    def onStartStop(self):
        if self.button['text'] == 'Start':
            self.game_active = True
            self.button.config(text="Stop")
            t1 = threading.Thread(target = self.run_game) 
            t1.start()
            
        else:
            self.game_active = False
            self.button.config(text="Start")
            t1.join()
        
        