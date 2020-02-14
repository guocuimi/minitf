from .. import kernal as K


class Tensor(object):
    def __init__(self, value):
        self._value = value

    data = property(lambda self: self._value)

    def __neg__(self): return K.negative(self)

    def __add__(self, other): return K.add(self, other)

    def __sub__(self, other): return K.subtract(self, other)

    def __mul__(self, other): return K.multiply(self, other)

    def __truediv__(self, other): return K.divide(self, other)

    def __radd__(self, other): return K.add(other, self)

    def __rsub__(self, other): return K.subtract(other, self)

    def __rmul__(self, other): return K.multiply(other, self)

    def __rtruediv__(self, other): return K.divide(other, self)

    def update_sub(self, dec): self._value -= get_val(dec)

    def numpy(self):
        return K.asnumpy(self._value)


def is_tensor(x):
    return type(x) in [Tensor]


def get_val(x):
    return x.data if is_tensor(x) else x
