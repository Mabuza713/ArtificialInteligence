from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np

class Perceptron(ABC):
    def __init__(self, weights, bias, learning_rate):
        self.initial_weights = weights
        self.initial_bias = bias

        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

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

            for x, y in zip(X, Y):
                y_pred = self.activation_function(x)
                error = y - y_pred
                if error != 0:
                    print(x, y, error, y_pred)
                    self.weights = [self.weights[index] + self.learning_rate * error * x[index] for index in range(len(self.weights))]
                    self.bias = self.bias +self.learning_rate * error

            if iteration % 20000 == 0 or (temp_weights == self.weights and self.bias == temp_bias):
                print('Iteracja: ', iteration)
                print(f"Wagi w iteracji [poprzedniej - teraźniejszej]: {temp_weights} - {self.weights}")
                print(f"obciążenie w iteracji [poprzedniej - teraźniejszej]: {temp_bias} - {self.bias}")
                print(f"Początkowe wartości wag: {self.initial_weights}")
                print(f"Początkowe wartości obciążenia: {self.initial_bias}")
                print("-------------------------------------------")
                print()
                if temp_weights == self.weights and self.bias == temp_bias:
                    break





class UnipolarPerceptron(Perceptron):
    def activation_function(self, X):
        return 1 if self.weighted_sum(X) > 0 else 0


"""
W perceptronie bipolarnym funkcja aktywacji zwraca dwie wartości: 1 lub -1,
odróżnia to ją od perceptrony unipolarnego. Wprowadzenie wartości -1 pozwala na 
zwiększenie intensywności uczenia tzn. w momęcie kiedy błąd pomiędzy wartością 
przewidzianą y_pred a wartością prawdziwą y będzie różny wówczas podczas dostrajania wag
naszej sieci nie będzie nam to wyłącznie modyfikowało znaku ale również pozwoli na szybsze osiągnięcie
poprawnych wag. 
"""
class BipolarPerceptron(Perceptron):
    def activation_function(self, X):
        return 1 if self.weighted_sum(X) > 0 else -1


# STWORZONE Z POMOCĄ AI, GDZIE POSŁUŻYŁO TYLKO
# I WYŁĄCZNIE DO STWORZENIA WYKRESÓW WZORY ZOSTAŁY
# WYPROWADZONOE WŁASNORĘCZNIE
def create_plot(X, Y, perceptron, nazwa_operacji):
    title = ""
    if isinstance(perceptron, UnipolarPerceptron):
        title = "Unipolar perceptron"
    else:
        title = "Bipolar perceptron"

    if len(perceptron.weights) == 1:
        title = title + " z jednym wejściem"
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
        plt.axvline(x=-perceptron.initial_bias / perceptron.initial_weights[0], color='yellow',label=f'Funckja wytworzona z początkowych wartości')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        colors_array = ["red" if y == 1 else "blue" for y in Y]
        plt.scatter([temp[0] for temp in X], [temp[1] for temp in X], c=colors_array, marker='o')
        x_vals = np.linspace(-1,  1.25, 200)
        if perceptron.weights[1] != 0:

            #Przekształcenie 2.
            y_vals = -(perceptron.weights[0]*x_vals + perceptron.bias)/perceptron.weights[1]
            y_vals_ini = -(perceptron.initial_weights[0]*x_vals + perceptron.initial_bias)/perceptron.initial_weights[1]
            plt.plot(x_vals, y_vals, color='green', label=f'Granica Decyzyjna\nb={perceptron.bias},\n w[0]={perceptron.weights[0]},\n w[1]={perceptron.weights[1]}')
            plt.plot(x_vals, y_vals_ini, color='yellow', label=f'Funckja wytworzona z początkowych wartości')
        plt.legend()
        plt.ylim(-0.25, 1.25)
        plt.xlim(-0.25, 1.25)
        plt.grid(True)
        plt.show()