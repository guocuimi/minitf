import os
import sys

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
    from .numpy import *
elif _BACKEND == 'cupy':
    sys.stderr.write('Using cupy\n')
    try:
        from .cupy import *
    except ImportError:
        sys.stderr.write('Can not load cupy, using numpy instead.\n')
        _BACKEND = 'numpy'
        from .numpy import *
else:
    raise ValueError('Unable to import : ' + str(_BACKEND))


def backend():
    return _BACKEND
