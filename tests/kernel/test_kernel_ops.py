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
    check_gradients(tf.square, [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_exp(arg):
    check_gradients(tf.exp, [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_negative(arg):
    check_gradients(tf.negative, [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_transpose(arg):
    grads = gradients(tf.transpose, [arg])
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

    check_gradients(tf.minimum,
                    [tf.random.randn(4), tf.random.randn(4)])


def test_cast():
    vec = tf.random.normal((10, 20), 0, 200.0, dtype=tf.float64)
    assert vec.dtype == tf.float64

    casted_vec = tf.cast(vec, tf.int32)
    assert casted_vec.dtype == tf.int32

    grads = gradients(lambda x: tf.cast(x, tf.int32), [vec])
    assert len(grads) == 1
    assert grads[0].dtype == tf.dtype("float64")


def test_dot():
    mat1 = tf.random.randn(50, 11)
    mat2 = tf.random.randn(11, 40)
    # vect1 = tf.random.randn(10)
    # vect2 = tf.random.randn(11)
    fun = lambda x, y: tf.dot(x, y)
    check_gradients(fun, [mat1, mat2])
    # check_gradients(fun, [vect1, mat1])
    # check_gradients(fun, [vect2, mat2])
