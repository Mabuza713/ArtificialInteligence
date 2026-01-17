import tkinter as tk

"""
tkinter poznalem dzieki 
https://www.youtube.com/watch?v=BsIQ9CjWLco


lewy guzik to zmiana typu cella
prawy to zapis do pliku MAZE.TXT
space to wczytanie z pliku SOLVEDMAZE.TXT

"""



AMOUNT_OF_COLS = 25
AMOUNT_OF_ROWS = 25

class MazeEditor:
    def __init__(self):
        self.root = tk.Tk()

        self.grid = [["." for _ in range(AMOUNT_OF_COLS)] for _ in range(AMOUNT_OF_ROWS)]
        self.start = None
        self.finish = None

        self.canvas = tk.Canvas(self.root, width=AMOUNT_OF_COLS*20, height=AMOUNT_OF_ROWS*20)
        self.canvas.pack()

        self.draw_grid()

        self.canvas.bind("<Button-1>", self.handle_left_click)
        self.canvas.bind("<Button-3>", self.handle_right_click)
        self.root.bind("<space>", lambda event: self.load_maze())

    def draw_grid(self):
        for row in range(AMOUNT_OF_ROWS):
            for column in range(AMOUNT_OF_COLS):
                x1 = column * 20
                y1 = row * 20
                x2 = x1 + 20
                y2 = y1 + 20
                self.canvas.create_line(x1, y1, x2, y2)

                char = self.grid[row][column]
                color = "white"
                if char == "S":
                    color = "green"
                if char == "K":
                    color = "red"
                if char == "#":
                    color = "black"
                if char == "*":
                    color = "blue"

                self.canvas.create_rectangle(x1, y1,x2, y2, fill=color, outline="black")

    def handle_left_click(self, event):
        column = event.x // 20
        row = event.y // 20

        if 0 <= column < AMOUNT_OF_COLS and 0 <= row and row < AMOUNT_OF_ROWS:
            if self.grid[row][column] == ".":
                self.grid[row][column] = "#"
                self.start = (column, row)
            elif self.grid[row][column] == "#":
                self.grid[row][column] = "S"
                self.finish = (column, row)
            elif self.grid[row][column] == "S":
                self.grid[row][column] = "K"
            else:
                self.grid[row][column] = "."


            self.draw_grid()

    def handle_right_click(self, event):
        print("dsds")
        with open("Maze.txt", "w") as file:
            for row in self.grid:
                file.write("".join(row) + "\n")

    def load_maze(self):
        with open("SolvedMaze.txt", "r") as file:
            file_lines = file.readlines()
            for row, line in enumerate(file_lines):
                self.grid[row] = list(line.strip())

        self.draw_grid()


maze = MazeEditor()
maze.root.mainloop()