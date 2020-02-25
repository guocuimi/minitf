from __future__ import absolute_import

import minitf as tf


def relu(x):
    return tf.maximum(0, x)


def leakyrelu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


def sigmoid(x):
    return 1.0 / (1 + tf.exp(-x))


def tanh(x):
    y = tf.exp(-x)
    return (1.0 - y) / (1.0 + y)


def linear(x):
    return x


def get(identifier):
    if identifier is None:
        return linear

    if isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError('Unknown activation function: {}'.format(identifier))
        return cls()
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret activation function identifier: ' +
                         str(identifier))
