import pytest

import minitf as tf
from minitf.test_util import check_gradients
from minitf.test_util import gradients

tf.random.set_seed(10)


def gen_args():
    scalar = tf.random.randn()
    vector = tf.random.randn(4)
    mat = tf.random.randn(3, 4)
    mat2 = tf.random.randn(1, 4)
    return [scalar, vector, mat, mat2]


# ------- Binary ops -------
@pytest.mark.parametrize('arg_x', gen_args())
@pytest.mark.parametrize('arg_y', gen_args())
def test_add(arg_x, arg_y):
    check_gradients(lambda x, y: x + y, [arg_x, arg_y])


@pytest.mark.parametrize('arg_x', gen_args())
@pytest.mark.parametrize('arg_y', gen_args())
def test_subtract(arg_x, arg_y):
    check_gradients(lambda x, y: x - y, [arg_x, arg_y])


@pytest.mark.parametrize('arg_x', gen_args())
@pytest.mark.parametrize('arg_y', gen_args())
def test_multiple(arg_x, arg_y):
    check_gradients(lambda x, y: x * y, [arg_x, arg_y])


@pytest.mark.parametrize('arg_x', gen_args())
@pytest.mark.parametrize('arg_y', gen_args())
def test_divide(arg_x, arg_y):
    check_gradients(lambda x, y: x / y, [arg_x, arg_y], atol=1e-2, rtol=1e-2)


# ------- Unary ops -------
@pytest.mark.parametrize('arg', gen_args())
def test_square(arg):
    check_gradients(lambda x: tf.square(x), [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_exp(arg):
    check_gradients(lambda x: tf.exp(x), [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_negative(arg):
    check_gradients(lambda x: tf.negative(x), [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_transpose(arg):
    grads = gradients(lambda x: tf.transpose(x), [arg])
    assert len(grads) == 1
    assert grads[0] == tf.ones_like(grads[0])


def test_maximum():
    check_gradients(lambda x: tf.maximum(x, x),
                    [tf.random.randn(4)])

    check_gradients(lambda x, y: tf.maximum(x, y),
                    [tf.random.randn(4), tf.random.randn(4)])


def test_minimum():
    check_gradients(lambda x: tf.minimum(x, x),
                    [tf.random.randn(4)])

    check_gradients(lambda x, y: tf.minimum(x, y),
                    [tf.random.randn(4), tf.random.randn(4)])
