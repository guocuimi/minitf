from minitf import kernel as K
from minitf.jvps.jvp_maker import def_jvp


# Stolen from autograd library
def unbroadcast(target, g):
    while K.rank(g) > K.rank(target):
        g = K.sum(g, axis=0)
    for axis, size in enumerate(K.shape(target)):
        if size == 1:
            g = K.sum(g, axis=axis, keepdims=True)
    return g


def balanced_eq(x, z, y):
    return (x == z) / (1.0 + (x == y))


def_jvp(K.add, lambda ans, x, y: (
    lambda g: unbroadcast(x, g),
    lambda g: unbroadcast(y, g),
))

def_jvp(K.subtract, lambda ans, x, y: (
    lambda g: unbroadcast(x, g),
    lambda g: unbroadcast(y, -g),
))

def_jvp(K.multiply, lambda ans, x, y: (
    lambda g: unbroadcast(x, y * g),
    lambda g: unbroadcast(y, x * g),
))

def_jvp(K.divide, lambda ans, x, y: (
    lambda g: unbroadcast(x, g / y),
    lambda g: unbroadcast(y, -g * x / (y * y)),
))

def_jvp(K.dot, lambda ans, x, y: (
    lambda g: K.dot(g, K.transpose(y)),
    lambda g: K.dot(K.transpose(x), g),
))

def_jvp(K.square, lambda ans, x: (
    lambda g: g * 2 * x,
))

def_jvp(K.reduce_mean, lambda ans, x: (
    lambda g: g / K.size(x),
))

def_jvp(K.exp, lambda ans, x: (
    lambda g: ans * g,
))

def_jvp(K.negative, lambda ans, x: (
    lambda g: -g,
))

def_jvp(K.transpose, lambda ans, x: (
    lambda g: K.transpose(g),
))

def_jvp(K.maximum, lambda ans, x, y: (
    lambda g: g * balanced_eq(x, ans, y),
    lambda g: g * balanced_eq(y, ans, x),
))

def_jvp(K.minimum, lambda ans, x, y: (
    lambda g: g * balanced_eq(x, ans, y),
    lambda g: g * balanced_eq(y, ans, x),
))
