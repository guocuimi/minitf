import numpy as _np

from minitf.kernel.core import notrace_primitive
from minitf.kernel.core import primitive

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
maximum = primitive(_np.maximum)
minimum = primitive(_np.minimum)

# temporarily put it here as nograd function
reduce_sum = notrace_primitive(_np.sum)
