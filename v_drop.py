from keras import backend as K
from keras import constraints
from keras.engine.topology import Layer
from keras.layers import Activation, Dropout, Dense, Input
from keras.models import Model
from keras import metrics
from keras import optimizers

import tensorflow as tf
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import math_ops

# some magic numbers
c1 = 1.16145124
c2 = -1.50204118
c3 = 0.58629921


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


def define_networks(input_shape,
                    n_hidden=4, s_hidden=64,
                    optimiser=optimizers.Adam(), magicrescale=1):

    '''
    '''

    x = Input(batch_shape=(input_shape), name='x_input')
    for i in range(n_hidden):
        name_layer_v = 'h' + str(i)
        name_layer_a = 'a' + str(i)
        if i == 0:
            vd = VariationalDropoutLayer(s_hidden, name=name_layer_v)(x)
        else:
            vd = VariationalDropoutLayer(s_hidden, name=name_layer_v)(net)
        net = Activation('relu', name=name_layer_a)(vd)

    output = Dense(10, name='o', activation='softmax')(net)

    my_model = Model(x, output)

    def customloss(y_true, y_pred):
        xent_loss = metrics.categorical_crossentropy(y_true, y_pred)

        klosslist = []
        for l in my_model.layers:
            if isinstance(l, VariationalDropoutLayer):
                a = l.alpha
                asq = a * a
                acu = asq * a
                klosslist.append(K.mean((.5 * K.log(a) + c1 *
                                         a + c2 * asq + c3 * acu)))

        return K.mean(xent_loss - magicrescale * K.sum(klosslist))

    my_model.compile(optimizer=optimiser, loss=customloss,
                     metrics=['categorical_accuracy'])

    return(my_model)
