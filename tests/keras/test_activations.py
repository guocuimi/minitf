import pytest

import minitf as tf
from minitf.keras import activations
from minitf.test_util import check_gradients

tf.random.set_seed(10)


def gen_args():
    scalar = tf.random.randn()
    vector = tf.random.randn(4)
    mat = tf.random.randn(3, 4)
    mat2 = tf.random.randn(10, 4)
    return [scalar, vector, mat, mat2]


@pytest.mark.parametrize('arg', gen_args())
def test_relu(arg):
    check_gradients(activations.relu, [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_leakyrelu(arg):
    check_gradients(activations.leakyrelu, [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_sigmoid(arg):
    check_gradients(activations.sigmoid, [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_tanh(arg):
    check_gradients(activations.tanh, [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_linear(arg):
    check_gradients(activations.linear, [arg])
