import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from keras import constraints
def get_mnist():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    y_train_1hot = to_categorical(y_train, 10)
    y_test_1hot = to_categorical(y_test, 10)

    return(x_train, x_test, y_train, y_test, y_train_1hot, y_test_1hot)


class alphaclip(constraints.Constraint):

    def __call__(self, w):
        return K.clip(w, 1e-8, .5)
        # we will log the weights in the loss computation, so we have to clip to epsilon
