from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

mnist = input_data.read_data_sets('/data/mnist/', one_hot=True)


def show_pics(data, num):
    pixels = data.train.images[num]
    pixels = pixels.reshape((28, 28))
    true_label = np.argmax(data.train.labels[num])

    plt.imshow(pixels, cmap=cm.Greys)
    plt.title("True Label is %d"%true_label)
    plt.show()
