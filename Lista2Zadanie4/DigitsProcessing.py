from matplotlib import pyplot as plt
from PIL import Image
import math

def load_dataset(dataset_directory):
    # Array of vectors representing digits
    x_array = []

    # Array of what given picture should depict
    y_array = []

    # Loading data set and simple processing
    with open(dataset_directory, "r") as textFile:
        for line in textFile.readlines():
            digit_array = line.split(",")[:64]
            label = line.split(",")[64]

            x_array.append([int(x) for x in digit_array])
            y_array.append([int(label)])

    return x_array, y_array


def load_data_from_image(image_directory, realLabel):
    # Array of vectors representing digits
    x_array = []

    # Array of what given picture should depict
    y_array = []
    final_array = []

    # Loading data set and simple processing
    img = Image.open(image_directory).convert("RGB")
    for x in range(0, 8):
        for y in range(0, 8):
            final_array.append(1 if sum(img.getpixel((y, x))) < 200 else 0)

    return final_array, [[realLabel]]

load_data_from_image("test.png", 2)

def plot_number_using_vector(pixelValues, realLabel):
    final_array = []
    for _ in range(0, 8):
        final_array.append([])

    for x in range(0, 8):
        for y in range(0, 8):
            final_array[y].append(int(pixelValues[x + 8 * y]))

    plt.imshow(final_array, cmap='binary', interpolation='nearest')
    plt.title(f"Image of {realLabel}")
    plt.show()
