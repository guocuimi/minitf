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


@pytest.mark.parametrize('arg_x', gen_args())
@pytest.mark.parametrize('arg_y', gen_args())
def test_maximum(arg_x, arg_y):
    check_gradients(lambda x: tf.maximum(x, x), [arg_x])
    check_gradients(lambda x: tf.maximum(x, x), [arg_y])
    check_gradients(tf.maximum, [arg_x, arg_y])


@pytest.mark.parametrize('arg_x', gen_args())
@pytest.mark.parametrize('arg_y', gen_args())
def test_minimum(arg_x, arg_y):
    check_gradients(lambda x: tf.minimum(x, x), [arg_x])
    check_gradients(lambda x: tf.minimum(x, x), [arg_y])
    check_gradients(tf.maximum, [arg_x, arg_y])

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
def test_flatten(arg):
    check_gradients(tf.flatten, [arg])


def test_flatten_basic():
    arg = tf.random.randn(3, 4, 5)
    check_gradients(tf.flatten, [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_reshape(arg):
    check_gradients(lambda x: tf.reshape(x, (-1)), [arg])


def test_reshape_basic():
    arg = tf.random.randn(3, 4, 5)
    check_gradients(lambda x: tf.reshape(x, (5, 12)), [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_transpose(arg):
    check_gradients(tf.transpose, [arg])


@pytest.mark.parametrize('arg', gen_args())
def test_where(arg):
    check_gradients(lambda x: tf.where(x > 0, x, x), [arg])
    check_gradients(lambda x: tf.where(x > 0, x, x * 2), [arg])


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
    check_gradients(tf.dot, [mat1, mat2])
    # check_gradients(fun, [vect1, mat1])
    # check_gradients(fun, [vect2, mat2])
