from __future__ import absolute_import

import os
import sys

from minitf.kernel.core import primitive

# Default backend: TensorFlow.
_BACKEND = 'numpy'

# Set backend based on KERAS_BACKEND flag, if applicable.
if 'MINITF_BACKEND' in os.environ:
    _backend = os.environ['MINITF_BACKEND']
    if _backend:
        _BACKEND = _backend

# Import backend functions.
if _BACKEND == 'numpy':
    sys.stderr.write('Using numpy\n')
    from minitf.kernel.numpy import *
    from minitf.kernel.numpy_math import *
    from minitf.kernel import numpy_math as math
    from minitf.kernel import numpy_random as random
elif _BACKEND == 'cupy':
    sys.stderr.write('Using cupy\n')
    try:
        from minitf.kernel.cupy import *
    except ImportError:
        sys.stderr.write('Can not load cupy, using numpy instead.\n')
        _BACKEND = 'numpy'
        from minitf.kernel.numpy import *
        from minitf.kernel.numpy_math import *
        from minitf.kernel import numpy_math as math
        from minitf.kernel import numpy_random as random
else:
    raise ValueError('Unable to import : ' + str(_BACKEND))


def backend():
    return _BACKEND


del absolute_import
