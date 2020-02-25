import math

import pytest

import minitf as tf
from minitf.keras import initializers

tf.random.set_seed(10)

_shapes = [
    (100,),
    (100, 2),
    (100, 2, 3),
    (100, 2, 3, 4),
    (100, 2, 3, 4, 5),
]


def _runner(init, shape, target_mean=None, target_std=None,
            target_max=None, target_min=None):
    output = init(shape).numpy()
    lim = 3e-1
    if target_std is not None:
        assert abs(output.std() - target_std) < lim
    if target_mean is not None:
        assert abs(output.mean() - target_mean) < lim
    if target_max is not None:
        assert abs(output.max() - target_max) < lim
    if target_min is not None:
        assert abs(output.min() - target_min) < lim


@pytest.mark.parametrize('shape', _shapes)
def test_uniform(shape):
    _runner(initializers.RandomUniform(minval=-1, maxval=1), shape,
            target_mean=0., target_max=1, target_min=-1)


@pytest.mark.parametrize('shape', _shapes)
def test_normal(shape):
    _runner(initializers.RandomNormal(mean=0, stddev=1), shape,
            target_mean=0., target_std=1)


@pytest.mark.parametrize('shape', _shapes)
def test_glorot_uniform(shape):
    fan_in, fan_out = initializers._compute_fans(shape)
    std = math.sqrt(2. / (fan_in + fan_out))
    _runner(initializers.glorot_uniform(), shape,
            target_mean=0., target_std=std)


@pytest.mark.parametrize('shape', _shapes)
def test_zero(shape):
    _runner(initializers.zeros(), shape,
            target_mean=0., target_max=0.)


@pytest.mark.parametrize('shape', _shapes)
def test_one(shape):
    _runner(initializers.ones(), shape,
            target_mean=1., target_max=1.)
