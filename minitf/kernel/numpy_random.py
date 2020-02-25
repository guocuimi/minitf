import numpy.random as _npr

from minitf.kernel.core import notrace_primitive

set_seed = _npr.seed
randn = notrace_primitive(_npr.randn)


@notrace_primitive
def normal(shape, mean=0.0, stddev=1.0, dtype=None):
    return _npr.normal(mean, stddev, shape)


@notrace_primitive
def uniform(shape, minval=0, maxval=None, dtype=None):
    return _npr.uniform(minval, maxval, shape)
