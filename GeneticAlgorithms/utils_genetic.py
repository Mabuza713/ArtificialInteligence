import random


"""
Problem odnajdywania najszybszej trasy, ogolnie mozna rozwiazac ten problem przy uzyciu np. algorytmu Djikstry
lub A*, 

moves = dna, na prezentacji jest wyrażona w sposób binarny, tzn. 1 lub 0, dla klasycznego problemu, w struktórze rozwiązania
bardziej będzie podobny do traveling salesman problemu. W traveling salesman mieslimy pokolei odwiedzane miasta, tutaj bedzie w ktora strone przesunac
ludka
"""

# Parametry ewolucji
MUTATION_RATE = 0.05
MAX_MOVES = 100
PASSTRHOUGH_LENGTH = 50


POSSIBLE_DIRECTIONS = {
    "GORA": (0, -1),
    "DOL": (0, 1),
    "LEWO": (-1, 0),
    "PRAWO": (1, 0)
}


# Parametry MAPY gdzie:
# . - puste pole przez ktore mozna przejsc
# # - przeszkode
# S - poczatek labiryntu
# K - koniec labiryntu
# * - droga rozwiazania



class Maze:
    def __init__(self):
        self.maze_map = []
        finish = (0, 0)
        start = (0, 0)

        with open("Maze.txt", "r") as file:
            file_lines = file.readlines()
            for row, line in enumerate(file_lines):
                self.maze_map.append(list(line.strip()))
                if "K" in self.maze_map[row]:
                    finish = (row, self.maze_map[row].index("K"))
                if "S" in self.maze_map[row]:
                    start = (row, self.maze_map[row].index("S"))


        self.start = {"x": start[0], "y": start[1]}
        self.finish = {"x": finish[0], "y": finish[1]}
        self.amount_of_rows = len(self.maze_map)
        self.amount_of_columns = len(self.maze_map[0])


class Traveler:
    def __init__(self, moves=None, maze=None):
        if moves is None:
            self.moves = [random.choice(list(POSSIBLE_DIRECTIONS.keys())) for _ in range(MAX_MOVES)]
        else:
            self.moves = moves

        self.current_pos = {"x": maze.start["x"], "y": maze.start["y"]}

        self.path = [(maze.start["x"], maze.start["y"])]
        self.finished = False
        self.maze = maze
        self.evaulation_value = 0
        # test


    def simulate(self):
        x = self.current_pos["x"]
        y = self.current_pos["y"]

        for move in self.moves:
            move_direction = POSSIBLE_DIRECTIONS[move]
            new_x = x + move_direction[0]
            new_y = y + move_direction[1]

            if (0 <= new_x < self.maze.amount_of_columns) and (0 <= new_y < self.maze.amount_of_rows) and self.maze.maze_map[new_y][new_x] != "#":
                x = new_x
                y = new_y
                self.current_pos = {"x": x, "y": y}
                self.path.append((x, y))
                if x == self.maze.finish["x"] and y == self.maze.finish["y"]:
                    self.finished = True
                    break

        self.current_pos = {"x": x, "y": y}

    # Chcemy jak najmniejszy dystans
    def evaluate(self):
        score = 0
        dist = abs(self.current_pos["x"] - self.maze.finish["x"]) + abs(self.current_pos["y"] - self.maze.finish["y"])

        score =- dist
        score = score -  len(self.path) * 0.01

        if self.finished :
            score += 1000
        self.evaulation_value = score

        return score


    def mutation(self):
        for i in range(len(self.moves)):
            if random.random() < MUTATION_RATE:
                self.moves[i] = random.choice(list(POSSIBLE_DIRECTIONS.keys()))



class PopulationManager:
    def __init__(self, population_size, maze, elite_variable, generation_split = None):
        self.current_population = [Traveler(maze=maze) for _ in range(population_size)]
        self.best_traveler = None
        self.maze = maze

        # Mowi ile najlepszych do next iteracji
        self.elite_variable = elite_variable

        self.generation_split = len(self.current_population) // 2 if generation_split is None else generation_split

    @staticmethod
    def crossing_over_single_point(traveler1, traveler2):
        crossing_over_point = random.randint(0, MAX_MOVES - 1)
        new_moves = traveler1.moves[:crossing_over_point] + traveler2.moves[crossing_over_point:]
        return Traveler(new_moves, maze=traveler1.maze)


    @staticmethod
    def crossing_over_two_points(t1, t2):
        pass



    def draw_maze(self, traveler):
        view = [row[:] for row in self.maze.maze_map]

        for (x, y) in traveler.path:
            if view[y][x] not in ["S", "K"]:
                view[y][x] = "*"

        print("\n" + "-"*20)
        for row in view:
            print("".join(row))
        print("-"*20)

        with open("SolvedMaze.txt", "w") as file:
            for row in view:
                file.write("".join(row) + "\n")

    def find_path(self, use_elitism=True, use_roulette=True, use_ranking=True):
        for generation in range(10000):
            for traveler in self.current_population:
                traveler.simulate()

            self.current_population.sort(key=lambda t: t.evaluate(), reverse=True)
            best_traveler = self.current_population[0]
            if generation % 1 == 0 or best_traveler.finished:
                print(f"Pokolenie nr: {generation}, najlepszy wynik: {best_traveler.evaluate()}")
                self.draw_maze(best_traveler)
                if best_traveler.finished and len(best_traveler.path) <= PASSTRHOUGH_LENGTH:
                    print("CEL OSIAGNIETY")
                    break

            new_population = []


            # DO SPRAWDZENIA TO JEST
            # elityzm mozna dodac kilu pierwszy np.3
            if use_elitism:
                new_population = self.current_population[ :self.elite_variable]

            if use_roulette:
                scores = [t.evaulation_value for t in self.current_population]
                min_score = min(scores)

                weights = [(score - min_score) / sum(scores)  + 1for score in scores]
                new_population.extend(random.choices(self.current_population, weights=weights, k=1))

            if use_ranking:
                k = 3
                q = 0
                for i in range(k):
                    for j in range(0, k - q):
                        new_population.append(self.current_population[q])
                    q += 1


            # Wybieramy tylko najlepszych do przekazania cech
            parents = self.current_population[ :self.generation_split]

            while len(new_population) < len(self.current_population):
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

                child = self.crossing_over_single_point(parent1, parent2)
                child.mutation()

                new_population.append(child)

            self.current_population = new_population





population_manager = PopulationManager(25, Maze(), 1)
population_manager.find_path(use_ranking = True)