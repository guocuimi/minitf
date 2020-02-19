import numpy as _np

from ..autodiff import (
    primitive,
    notrace_primitive,
)

# ----- Non-differentiable functions -----
rank = notrace_primitive(_np.ndim, as_tensor=False)
shape = notrace_primitive(_np.shape, as_tensor=False)

# Always return Tensor object
size = notrace_primitive(_np.size)
zeros_like = notrace_primitive(_np.zeros_like)
ones_like = notrace_primitive(_np.ones_like)
linspace = notrace_primitive(_np.linspace)

# temporarily put it here as nograd function
sum = notrace_primitive(_np.sum)


@notrace_primitive
def constant(value, dtype=None, shape=None):
    t = _np.array(value, dtype=dtype)
    if shape is None or _np.shape(t) == shape:
        return t

    num_t = _np.size(t)
    expect_size = 1
    for dim in shape:
        expect_size *= dim

    if num_t == expect_size:
        return _np.reshape(t, shape)
    if num_t == 1:
        return _np.full(shape, value)
    raise TypeError("tf.constant with unsupported shape "
                    "(value has %d elements, shape is %s with %d elements)." %
                    (num_t, shape, expect_size))


transpose = primitive(_np.transpose)


def asnumpy(x):
    return x
