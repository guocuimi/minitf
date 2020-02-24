import numpy.random as _npr

from minitf.kernel.core import notrace_primitive

set_seed = _npr.seed
normal = notrace_primitive(_npr.normal)
randn = notrace_primitive(_npr.randn)
