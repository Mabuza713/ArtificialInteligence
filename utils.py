from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np

class Perceptron(ABC):
    def __init__(self, weights, bias, learning_rate,dir_name = ""):
        self.initial_weights = weights
        self.initial_bias = bias

        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

        self.dir_name = dir_name

    def reset_weights(self):
        self.weights = self.initial_weights
        self.bias = self.initial_bias

    def weighted_sum(self, X):
        return sum([x * w for x, w in zip(self.weights, X)]) + self.bias


    @abstractmethod
    def activation_function(self, x):
        return NotImplemented


    def tune_weights(self, X, Y, amount_of_iterations):
        if isinstance(self, BipolarPerceptron):
            Y = [-1 if y == 0 else 1 for y in Y]

        for iteration in range(amount_of_iterations):
            temp_weights = self.weights[:]
            temp_bias = self.bias
            no_change = True
            for x, y in zip(X, Y):
                y_pred = self.activation_function(x)
                error = y - y_pred
                if error != 0:
                    no_change = False
                    self.weights = [self.weights[index] + self.learning_rate * error * x[index] for index in range(len(self.weights))]
                    self.bias = self.bias +self.learning_rate * error

            if iteration % 10 == 0 or no_change:
                print(no_change)
                print('Iteracja: ', iteration)
                print(f"Wagi w iteracji [poprzedniej - teraźniejszej]: {temp_weights} - {self.weights}")
                print(f"obciążenie w iteracji [poprzedniej - teraźniejszej]: {temp_bias} - {self.bias}")
                print(f"Początkowe wartości wag: {self.initial_weights}")
                print(f"Początkowe wartości obciążenia: {self.initial_bias}")
                print("-------------------------------------------")
                print()
                create_plot(X, Y, self, "temp", f"{self.dir_name}/{iteration}")
                if no_change:
                    break





class UnipolarPerceptron(Perceptron):
    def activation_function(self, X):
        return 1 if self.weighted_sum(X) > 0 else 0


class BipolarPerceptron(Perceptron):
    def activation_function(self, X):
        return 1 if self.weighted_sum(X) > 0 else -1


# STWORZONE Z POMOCĄ AI, GDZIE POSŁUŻYŁO TYLKO
# I WYŁĄCZNIE DO STWORZENIA WYKRESÓW WZORY ZOSTAŁY
# WYPROWADZONOE WŁASNORĘCZNIE
def create_plot(X, Y, perceptron, nazwa_operacji, name=None):
    title = ""
    if isinstance(perceptron, UnipolarPerceptron):
        title = "Unipolar perceptron"
    else:
        title = "Bipolar perceptron"

    if len(perceptron.weights) == 1:
        title = title + " z jednym wejściem"
    elif len(perceptron.weights) == 5:
        title = title + "z pięcioma wejściami"
    else:
        title = title + " z dwoma wejściami"
    title = title + f"\ndla operacji {nazwa_operacji}"
    plt.title(title)
    if len(perceptron.weights) == 1:
        colors_array = ["red" if y == 1 else "blue" for y in Y]
        plt.scatter([temp[0] for temp in X], [0 for z in range(len(Y))],c=colors_array, marker='o')

        #Przekształcenie 1.
        boundry_x = -perceptron.bias / perceptron.weights[0]

        plt.axvline(x=boundry_x, color='green', label=f'Granica Decyzyjna\nb={perceptron.bias}, w[0]={perceptron.weights[0]}')
        plt.axvline(x=-perceptron.initial_bias / perceptron.initial_weights[0], color='cyan',linestyle="--",label=f'Funckja wytworzona z początkowych wartości')
        plt.legend()
        plt.grid(True)
        if name:
            plt.savefig(f"{name}.jpg")
            plt.close()
        plt.show()
    elif len(perceptron.weights) == 5:
        colors_array = ["red" if y == 1 else "blue" for y in Y]
        plt.scatter([temp[0] for temp in X], [temp[1] for temp in X], c=colors_array, marker='o')

        ax_lim = 1.5
        x_vals_grid = np.linspace(-ax_lim, ax_lim, 200)
        y_vals_grid = np.linspace(-ax_lim, ax_lim, 200)
        X_grid, Y_grid = np.meshgrid(x_vals_grid, y_vals_grid)


        boundry = (perceptron.weights[0] * X_grid ** 2 + perceptron.weights[1] * Y_grid ** 2 + perceptron.weights[2] * X_grid * Y_grid +
             perceptron.weights[3] * X_grid + perceptron.weights[4] * Y_grid + perceptron.bias)

        boundry_initial = (perceptron.initial_weights[0] * X_grid ** 2 + perceptron.initial_weights[1] * Y_grid ** 2 + perceptron.initial_weights[2] * X_grid * Y_grid +
                 perceptron.initial_weights[3] * X_grid + perceptron.initial_weights[4] * Y_grid + perceptron.initial_bias)

        plt.contour(X_grid, Y_grid, boundry, levels=[0], colors='green')
        plt.contour(X_grid, Y_grid, boundry_initial, levels=[1], colors='cyan', linestyles='--')


        plt.ylim(-1.5, 1.5)
        plt.xlim(-1.5, 1.5)
        plt.grid(True)
        if name:
            plt.savefig(f"{name}.jpg")
            plt.close()
        plt.show()


    else:
        colors_array = ["red" if y == 1 else "blue" for y in Y]
        plt.scatter([temp[0] for temp in X], [temp[1] for temp in X], c=colors_array, marker='o')
        x_vals = np.linspace(-1,  1.25, 200)
        if perceptron.weights[1] != 0:

            y_vals = -(perceptron.weights[0]*x_vals + perceptron.bias)/perceptron.weights[1]
            y_vals_ini = -(perceptron.initial_weights[0]*x_vals + perceptron.initial_bias)/perceptron.initial_weights[1]
            plt.plot(x_vals, y_vals, color='green', label=f'Granica Decyzyjna\nb={perceptron.bias},\n w[0]={perceptron.weights[0]},\n w[1]={perceptron.weights[1]}')
            plt.plot(x_vals, y_vals_ini, color='yellow',linestyle="--" , label=f'Funckja wytworzona z początkowych wartości')
        plt.legend()
        plt.ylim(-0.5, 1.5)
        plt.xlim(-0.5, 1.5)
        plt.grid(True)

        if name:
            plt.savefig(f"{name}.jpg")
            plt.close()
        plt.show()