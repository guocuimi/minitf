from minitf import kernel as K
from minitf.vjps.vjp_maker import def_vjp_maker


# Stolen from autograd library
def unbroadcast(target, g):
    while K.rank(g) > K.rank(target):
        g = K.reduce_sum(g, axis=0)
    for axis, size in enumerate(K.shape(target)):
        if size == 1:
            g = K.reduce_sum(g, axis=axis, keepdims=True)
    return g


def balanced_eq(x, z, y):
    return (x == z) / (1.0 + (x == y))


def_vjp_maker(K.add, lambda ans, x, y: (
    lambda g: unbroadcast(x, g),
    lambda g: unbroadcast(y, g),
))

def_vjp_maker(K.subtract, lambda ans, x, y: (
    lambda g: unbroadcast(x, g),
    lambda g: unbroadcast(y, -g),
))

def_vjp_maker(K.multiply, lambda ans, x, y: (
    lambda g: unbroadcast(x, y * g),
    lambda g: unbroadcast(y, x * g),
))

def_vjp_maker(K.divide, lambda ans, x, y: (
    lambda g: unbroadcast(x, g / y),
    lambda g: unbroadcast(y, -g * x / (y * y)),
))

def_vjp_maker(K.dot, lambda ans, x, y: (
    lambda g: K.dot(g, K.transpose(y)),
    lambda g: K.dot(K.transpose(x), g),
))

def_vjp_maker(K.square, lambda ans, x: (
    lambda g: g * 2 * x,
))

def_vjp_maker(K.reduce_mean, lambda ans, x: (
    lambda g: g / K.size(x),
))

def_vjp_maker(K.exp, lambda ans, x: (
    lambda g: ans * g,
))

def_vjp_maker(K.negative, lambda ans, x: (
    lambda g: -g,
))

def_vjp_maker(K.transpose, lambda ans, x: (
    lambda g: K.transpose(g),
))

def_vjp_maker(K.maximum, lambda ans, x, y: (
    lambda g: g * balanced_eq(x, ans, y),
    lambda g: g * balanced_eq(y, ans, x),
))

def_vjp_maker(K.minimum, lambda ans, x, y: (
    lambda g: g * balanced_eq(x, ans, y),
    lambda g: g * balanced_eq(y, ans, x),
))

def_vjp_maker(K.cast, lambda ans, x, dtype: (
    lambda g: K.cast(g, x.dtype),
))
