import numpy as _np

from minitf.kernel.core import notrace_primitive
from minitf.kernel.core import primitive

# ----- Export defined dtype -----
dtype = _np.dtype
float16 = _np.float16
float32 = _np.float32
float64 = _np.float64
int8 = _np.int8
int16 = _np.int16
int32 = _np.int32
int64 = _np.int64
uint8 = _np.uint8
uint16 = _np.uint16
uint32 = _np.uint32
uint64 = _np.uint64
bool = _np.bool_


# string = _np.object

# default dtype
def floatx():
    return float32

# ----- Non-differentiable functions -----
rank = notrace_primitive(_np.ndim, as_tensor=False)
shape = notrace_primitive(_np.shape, as_tensor=False)
allclose = notrace_primitive(_np.allclose, as_tensor=False)

# Always return Tensor object
size = notrace_primitive(_np.size)
zeros_like = notrace_primitive(_np.zeros_like)
ones_like = notrace_primitive(_np.ones_like)
linspace = notrace_primitive(_np.linspace)

# Compare ops
equal = notrace_primitive(_np.equal)
not_equal = notrace_primitive(_np.not_equal)
greater = notrace_primitive(_np.greater)
greater_equal = notrace_primitive(_np.greater_equal)
less = notrace_primitive(_np.less)
less_equal = notrace_primitive(_np.less_equal)

# temporarily put it here as nograd function
sum = notrace_primitive(_np.sum)

transpose = primitive(_np.transpose)


@primitive
def cast(x, dtype):
    return x.astype(dtype)


@notrace_primitive
def constant(value, dtype=None, shape=None):
    # already a numpy array, just return
    if isinstance(value, _np.ndarray):
        return value

    # construct numpy array with value and dtype
    t = _np.array(value, dtype=dtype)

    # cast down from 64 bits to 32 bits if necessary
    if dtype is None and t.dtype != floatx() and t.dtype == _np.float64:
        t = t.astype(floatx())

    if shape is None or _np.shape(t) == shape:
        return t

    num_t = _np.size(t)
    expect_size = _np.prod(shape)

    if num_t == expect_size:
        return _np.reshape(t, shape)

    if num_t == 1:
        return _np.full(shape, value)

    raise TypeError("tf.constant with unsupported shape "
                    "(value has %d elements, shape is %s with %d elements)." %
                    (num_t, shape, expect_size))


def asnumpy(x):
    return x
