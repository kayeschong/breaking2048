from tkinter import Frame, Button

class Controls(Frame):
    def __init__(self, root):
        Frame.__init__(self, root)
        self.expectimax = Button(self, text="Expectimax", width=6)
        self.expectimax.pack()
    