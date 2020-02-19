import numpy as _np

from ..autodiff import primitive

# ----- Differentiable functions -----
add = primitive(_np.add)
subtract = primitive(_np.subtract)
multiply = primitive(_np.multiply)
divide = primitive(_np.divide)
dot = primitive(_np.dot)
square = primitive(_np.square)
reduce_mean = primitive(_np.average)
exp = primitive(_np.exp)
negative = primitive(_np.negative)
