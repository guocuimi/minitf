import numpy.random as _npr

from minitf.kernel.core import notrace_primitive
from minitf.kernel._numpy import floatx

set_seed = _npr.seed
randn = notrace_primitive(_npr.randn)


@notrace_primitive
def normal(shape, mean=0.0, stddev=1.0, dtype=None):
    if dtype is None:
        dtype = floatx()
    return _npr.normal(mean, stddev, shape).astype(dtype)


@notrace_primitive
def uniform(shape, minval=0, maxval=None, dtype=None):
    if dtype is None:
        dtype = floatx()
    return _npr.uniform(minval, maxval, shape).astype(dtype)
