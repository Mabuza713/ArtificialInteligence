from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np

class Perceptron(ABC):
    def __init__(self, weights, bias, learning_rate):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate


    def weighted_sum(self, X):
        return sum([x * w for x, w in zip(self.weights, X)]) + self.bias


    @abstractmethod
    def activation_function(self, x):
        return NotImplemented


    def tune_weights(self, X, Y, amount_of_iterations):
        for iteration in range(amount_of_iterations):
            for x, y in zip(X, Y):
                y_pred = self.activation_function(x)
                if y_pred != y:
                    error = y - y_pred
                    self.weights = [self.weights[index] + self.learning_rate * error * x[index] for index in range(len(self.weights))]
                    self.bias = self.bias +self.learning_rate * error

        print(self.weights)


class UnipolarPerceptron(Perceptron):
    def activation_function(self, X):
        return 1 if self.weighted_sum(X) >= 0 else 0


"""
W perceptronie bipolarnym funkcja aktywacji zwraca dwie wartości: 1 lub -1,
odróżnia to ją od perceptrony unipolarnego. Wprowadzenie wartości -1 pozwala na
zwiększenie intensywności uczenia tzn. częstości modifikowania wag,

"""
class BipolarPerceptron(Perceptron):
    def activation_function(self, X):
        return 1 if self.weighted_sum(X) >= 0 else -1


# STWORZONE Z POMOCĄ AI
def create_plot(X, Y, perceptron):
    if len(perceptron.weights) == 1:
        colors_array = ["red" if y == 1 else "blue" for y in Y]
        plt.scatter([temp[0] for temp in X], [0 for z in range(len(Y))],c=colors_array, marker='o')


        plt.show()
    else:
        colors_array = ["red" if y == 1 else "blue" for y in Y]
        plt.scatter([temp[0] for temp in X], [temp[1] for temp in X], c=colors_array, marker='o')
        x_vals = np.linspace(-1, + 1, 200)
        if perceptron.weights[1] != 0:
            # Chcąc splotować
            y_vals = -(perceptron.weights[0]*x_vals + perceptron.bias)/perceptron.weights[1]
            plt.plot(x_vals, y_vals, color='green', label='decision boundary')
        plt.show()