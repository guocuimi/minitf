import numpy as _np

from ..autodiff import primitive

# ----- Non-differentiable functions -----
nograd_functions = [
    _np.ndim,
    _np.shape,
    _np.size,
    _np.zeros_like,
    _np.ones_like,
    _np.sum,  # temporarily put it here as nograd function

]

# ----- Differentiable functions -----
grad_functions = [
    _np.add,
    _np.subtract,
    _np.multiply,
    _np.divide,
    _np.dot,
    _np.square,
    _np.average,
]


def wrap_namespace(old, new):
    for name, obj in old.items():
        if obj in nograd_functions:
            new[name] = obj
        elif obj in grad_functions:
            new[name] = primitive(obj)


wrap_namespace(_np.__dict__, globals())
