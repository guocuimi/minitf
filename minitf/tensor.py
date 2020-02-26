from minitf import kernel as K


class Tensor(object):
    def __init__(self, value):
        # TODO: cast to numpy array
        self._value = get_val(value)

    def __neg__(self): return K.negative(self)

    def __add__(self, other): return K.add(self, other)

    def __sub__(self, other): return K.subtract(self, other)

    def __mul__(self, other): return K.multiply(self, other)

    def __truediv__(self, other): return K.divide(self, other)

    def __radd__(self, other): return K.add(other, self)

    def __rsub__(self, other): return K.subtract(other, self)

    def __rmul__(self, other): return K.multiply(other, self)

    def __rtruediv__(self, other): return K.divide(other, self)

    def __eq__(self, other): return K.equal(self, other)

    def __ne__(self, other): return K.not_equal(self, other)

    def __gt__(self, other): return K.greater(self, other)

    def __ge__(self, other): return K.greater_equal(self, other)

    def __lt__(self, other): return K.less(self, other)

    def __le__(self, other): return K.less_equal(self, other)

    def __hash__(self): return id(self)

    def __str__(self):
        numpy_obj = self.numpy()
        return "tf.Tensor(id=%s, shape=%s, dtype=%s, numpy=%s)" % (
            id(self), K.shape(numpy_obj), self.dtype, str(numpy_obj))

    def __repr__(self):
        numpy_obj = self.numpy()
        return "<tf.Tensor: id=%s, shape=%s, dtype=%s, numpy=%s>" % (
            id(self), K.shape(numpy_obj), self.dtype, numpy_obj)

    def numpy(self): return K.asnumpy(self._value)

    @property
    def data(self): return self._value

    @property
    def dtype(self): return getattr(self._value, 'dtype', None)


def is_tensor(x):
    return isinstance(x, Tensor)


def get_val(x):
    return x.data if is_tensor(x) else x