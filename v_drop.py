from keras import backend as K
from keras import constraints
from keras.engine.topology import Layer


import tensorflow as tf
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import math_ops


def my_init(shape, dtype=None):
    ''' a custom initializer for the initial
        weights of the variational dropout layer

    '''
    u = K.log(K.random_uniform(shape, minval=1e-8, maxval=.3, dtype=dtype))
    return(u)


class alphaclip(constraints.Constraint):

    def __call__(self, w):
        return K.clip(w, 1e-8, .99)
        # we will log the weights in the loss computation,
        # so we have to clip to epsilon


class VariationalDropoutLayer(Layer):
    # dropout rate by weight

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(VariationalDropoutLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        sti1 = input_shape[1]

        # Create a trainable weight variable for this layer.
        self.theta = self.add_weight(name='theta',
                                     shape=(sti1, self.output_dim),
                                     initializer='he_normal',
                                     trainable=True)
        self.alpha = self.add_weight(name='alpha',
                                     shape=(sti1, self.output_dim),
                                     initializer=my_init,
                                     trainable=True,
                                     constraint=alphaclip()
                                     )
        self.store_input = sti1
        super(VariationalDropoutLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, x, training=None):

        seed1, seed2 = random_seed.get_seed(123)

        def dropped_x():

            rnd = gen_random_ops._random_standard_normal(
                (self.store_input, self.output_dim),
                dtype=tf.float32,
                seed=seed1,
                seed2=seed2)
            mul = rnd * self.alpha
            si = math_ops.add(mul, tf.ones(
                (self.store_input, self.output_dim)), name='dropped')

            wi = si * self.theta
            b = K.dot(x, wi)
            return(b)

        c = K.dot(x, self.theta)

        return K.in_train_phase(dropped_x, c,
                                training=training)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
