from .. import kernal as K
from ..autodiff import def_jvp


def unbroadcast(target, g):
    while K.ndim(g) > K.ndim(target):
        g = K.sum(g, axis=0)
    for axis, size in enumerate(K.shape(target)):
        if size == 1:
            g = K.sum(g, axis=axis, keepdims=True)
    return g


def_jvp(K.add, lambda ans, x, y: (
    lambda g: unbroadcast(x, g),
    lambda g: unbroadcast(y, g)))

def_jvp(K.subtract, lambda ans, x, y: (
    lambda g: unbroadcast(x, g),
    lambda g: unbroadcast(y, -g)))

def_jvp(K.multiply, lambda ans, x, y: (
    lambda g: unbroadcast(x, y * g),
    lambda g: unbroadcast(y, x * g)))

def_jvp(K.divide, lambda ans, x, y: (
    lambda g: unbroadcast(x, g / y),
    lambda g: unbroadcast(y, -g * x / y ** 2)))

def_jvp(K.dot, lambda ans, x, y: (
    lambda g: K.dot(g, y.T),
    lambda g: K.dot(x.T, g)))

def_jvp(K.square, lambda ans, x: (
    lambda g: g * 2 * x,))

def_jvp(K.average, lambda ans, x: (
    lambda g: g / K.size(x),))
