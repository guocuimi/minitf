import math

import minitf as tf


def _compute_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        fan_in = math.sqrt(sum(shape))
        fan_out = math.sqrt(sum(shape))
    return fan_in, fan_out


class Initializer(object):
    def __call__(self, shape, dtype=None):
        raise NotImplementedError


class Zeros(Initializer):
    def __call__(self, shape, dtype=None):
        return tf.constant(0, shape=shape, dtype=dtype)


class Ones(Initializer):
    def __call__(self, shape, dtype=None):
        return tf.constant(1, shape=shape, dtype=dtype)


class RandomNormal(Initializer):
    def __init__(self, mean=0., stddev=0.05):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=None):
        return tf.random.normal(shape, self.mean, self.stddev, dtype=dtype)


class RandomUniform(Initializer):
    def __init__(self, minval=-0.05, maxval=0.05):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, shape, dtype=None):
        return tf.random.uniform(shape, self.minval, self.maxval, dtype=dtype)


class GlorotUniform(Initializer):
    """
    A uniform distribution within [-limit, limit] where limit is sqrt(6 / (fan_in + fan_out))
    """

    def __call__(self, shape, dtype=None):
        fan_in, fan_out = _compute_fans(shape)
        scale = 6.0 / max(1., (fan_in + fan_out))
        limit = math.sqrt(scale)
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)


# define alias for easier use
zero = zeros = Zeros
one = ones = Ones
uniform = random_uniform = RandomUniform
normal = random_normal = RandomNormal
xavier = xavier_uniform = glorot_uniform = GlorotUniform


def get(identifier):
    if isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError('Unknown initializer: {}'.format(identifier))
        return cls()
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret initializer identifier: ' +
                         str(identifier))
