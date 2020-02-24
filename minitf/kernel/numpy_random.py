import numpy.random as _npr

from ..autodiff import notrace_primitive

set_seed = _npr.seed
normal = notrace_primitive(_npr.normal)
randn = notrace_primitive(_npr.randn)
