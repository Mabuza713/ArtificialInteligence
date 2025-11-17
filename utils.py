from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np
import random
import math

class Perceptron(ABC):
    def __init__(self, weights, bias, learning_rate = 0,dir_name = ""):
        self.initial_weights = weights
        self.initial_bias = bias

        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.activation = 0

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

            if iteration % 1 == 0 or no_change:
                print(no_change)
                print('Iteracja: ', iteration)
                print(f"Wagi w iteracji [poprzedniej - teraźniejszej]: {temp_weights} - {self.weights}")
                print(f"obciążenie w iteracji [poprzedniej - teraźniejszej]: {temp_bias} - {self.bias}")
                print(f"Początkowe wartości wag: {self.initial_weights}")
                print(f"Początkowe wartości obciążenia: {self.initial_bias}")
                if no_change:
                    print(f"blad: {error}")
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


class Sigmoid(Perceptron):
    def __init__(self, weights, bias, learning_rate = 0, dir_name=""):
        super().__init__(weights, bias, learning_rate, dir_name)

    def activation_function(self, X):
        self.activation = 1 / (1 + np.exp(-self.weighted_sum(X)))
        return 1 / (1 + np.exp(-self.weighted_sum(X)))

    def derivative(self):
        return self.activation * (1 - self.activation)


class Relu(Perceptron):
    def __init__(self, weights, bias, learning_rate=0, dir_name=""):
        super().__init__(weights, bias, learning_rate, dir_name)
        self.activation = 0

    def activation_function(self, X):
        self.activation = max(0, self.weighted_sum(X))
        return self.activation

    # Troche naciagane bo nie da sie obliczyc z tego pochodnej ale sie tak przyjmuje
    def derivative(self):
        return 1 if self.activation > 0 else 0

class Softmax(Perceptron):
    def __init__(self, weights, bias, learning_rate = 0, dir_name=""):
        super().__init__(weights, bias, learning_rate, dir_name)
        self.exp_value = 0
        self.activation = 0


    # X tutaj stosowane jest tak jakby jak mamy wszystkie outputy w sensie prawdopodobienstwa
    # to bierzemy ich exp i dzielimy przez to, to jest wlasnie ten X
    def activation_function(self, X):
        self.activation = self.exp_value / X

    # Nie jest to pochodna softmaxa, ale to przez to że jak liczymy CCE to nie mnożymy razy
    # pochodną softmaxa, więc pominiecie tego stepu to poprostu pomnożenie razy 1
    def derivative(self):
        return 1


class NeuralNetwork:
    def __init__(self, hidden_layer, output_layer, learnign_rate):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.learning_rate = learnign_rate

    def reset_weights(self):
        for neuron in self.hidden_layer:
            neuron.weights = [random.uniform(-1, 1) for _ in neuron.weights]
            neuron.bias = random.uniform(-1, 1)

        for neuron in self.output_layer:
            neuron.weights = [random.uniform(-1, 1) for _ in neuron.weights]
            neuron.bias = random.uniform(-1, 1)

    # Majac jakis input, chcemy przewidziec jaka bedzie wartosc, tak jakby idziemy od lewej do
    # prawej w naszej sieci
    def forward_propagation(self, X):
        for perceptron in self.hidden_layer:
            perceptron.activation_function(X)

        if isinstance(self.output_layer[0], Softmax):
            output_weigted_sums = []
            for perceptron in self.output_layer:
                perceptron.exp_value = perceptron.weighted_sum([perceptron.activation for perceptron in self.hidden_layer])
                output_weigted_sums.append(perceptron.exp_value)

            for perceptron in self.output_layer:
                perceptron.exp_value = np.exp(perceptron.exp_value - max(output_weigted_sums))

            sum_of_exp = np.sum([perceptron.exp_value for perceptron in self.output_layer])
            for perceptron in self.output_layer:
                perceptron.activation_function(sum_of_exp)
        else:
            for perceptron in self.output_layer:
                perceptron.activation_function([x.activation for x in self.hidden_layer])

        return [perceptron.activation for perceptron in self.output_layer]

    # Jest to proces oceniania aktualnych wag w naszej sieci, najpierw przechodzimy
    # od lewej do prawej w celu uzyskania naszej predykcji, następnie porównujemy ją
    # z outputem z naszego zbioru uczącego i wrazie niezgodnosci korygujemy wagi idąc od
    # prawej do lewej tzn. zaczynając od warstwy wyjściowej
    def back_propagation(self, X, Y, verbose):
        old_weights = [output_perceptron.weights[:] for output_perceptron in self.output_layer]
        for index, output_perceptron in enumerate(self.output_layer):
            error =  output_perceptron.activation - Y[index]
            if verbose:
                print("-----")
                print(f"output perceptron {index}")
                print(f"\terror: {error}")
            for j, hidden_perceptron in enumerate(self.hidden_layer):
                new_weight = error * output_perceptron.derivative() * hidden_perceptron.activation
                output_perceptron.weights[j] -= new_weight * self.learning_rate
            output_perceptron.bias -= error * output_perceptron.derivative() * self.learning_rate


        for i, hidden_perceptron in enumerate(self.hidden_layer):
            if verbose:
                print("-----")
                print(f"hidden perceptron {i}")
                print("")

            sum_error = 0
            for j, output_perceptron in enumerate(self.output_layer):
                if verbose:
                    print(f"\toutput perceptron {j}")
                output_error = output_perceptron.activation - Y[j]
                if verbose:
                    print(f"\toutput error: {output_error}")


                output_error *= output_perceptron.derivative()
                if verbose:
                    print(f"\toutput der: {output_perceptron.derivative()}")

                output_error *= old_weights[j][i]
                if verbose:
                    print(f"\toutput old weight: {old_weights[j][i]}")
                    print("")
                sum_error += output_error
            hidden_error = sum_error * hidden_perceptron.derivative()
            if verbose:
                print(f"SUMA: {sum_error}")
                print(f"hidden der: {hidden_perceptron.derivative()}")
                print("")
            for k in range(0, len(X)):
                if verbose:
                    print(f"input index {k}")
                    print(f"\thidden error: {hidden_error * X[k]}")
                hidden_perceptron.weights[k] -= hidden_error * X[k] * self.learning_rate
            if verbose:
                print("------")
            hidden_perceptron.bias -= hidden_error * self.learning_rate



    def train(self, X, Y, amount_of_iterations, verbose = False):
        for e in range(amount_of_iterations):
            suma_bledu = 0
            for x, y in zip(X, Y):
                self.forward_propagation(x)
                if isinstance(self.output_layer[0], Softmax):
                    for index in range(len(y)):
                        suma_bledu -= math.log(self.output_layer[index].activation) * y[index]
                else:
                    for index in range(len(y)):
                        suma_bledu += 0.5 * (y[index] - self.output_layer[index].activation) ** 2
                self.back_propagation(x, y, verbose)
            print(f'iteracja {e}- wartość f.kosztu: {suma_bledu}')


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

def create_plot_with_two(X, Y, perceptrons):
    for perceptron in perceptrons:
        colors_array = ["red" if y == 1 else "blue" for y in Y]
        plt.scatter([temp[0] for temp in X], [temp[1] for temp in X], c=colors_array, marker='o')
        x_vals = np.linspace(-1,  1.25, 200)
        y_vals = -(perceptron.weights[0]*x_vals + perceptron.bias)/perceptron.weights[1]
        y_vals_ini = -(perceptron.initial_weights[0]*x_vals + perceptron.initial_bias)/perceptron.initial_weights[1]
        plt.plot(x_vals, y_vals, color='green')
        plt.plot(x_vals, y_vals_ini, color='yellow',linestyle="--" )
    plt.legend()
    plt.ylim(-0.5, 1.5)
    plt.xlim(-0.5, 1.5)
    plt.grid(True)
    plt.show()

#do formatowanie wartosci np na decimals w liscie
def format_list(lst, decimals=2):
    return [round(float(x), decimals) for x in lst]

def one_hot_encode(y):
    n_classes = np.max(y) + 1
    one_hot = np.zeros((np.array(y).shape[0], n_classes))
    for i, val in enumerate(y):
        one_hot[i, val[0]] = 1
    return one_hot

def acc_check(y_pred, y_true, acc):
    pred_class = np.argmax(y_pred)
    true_class = np.argmax(y_true)

    if pred_class == true_class:
        acc.append(1)
