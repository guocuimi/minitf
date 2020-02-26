from minitf import kernel as K
from minitf.tensor import Tensor
from minitf.tensor import get_val


class Variable(Tensor):
    def __init__(self, value):
        # TODO: cast to numpy array
        super(Variable, self).__init__(get_val(value))

    def __str__(self):
        numpy_obj = self.numpy()
        return "tf.Variable(id=%s, shape=%s, dtype=%s, numpy=%s)" % (
            id(self), K.shape(numpy_obj), self.dtype, str(numpy_obj))

    def __repr__(self):
        numpy_obj = self.numpy()
        return "<tf.Variable: id=%s, shape=%s, dtype=%s, numpy=%s>" % (
            id(self), K.shape(numpy_obj), self.dtype, numpy_obj)

    def assign_sub(self, delta):
        self._value -= get_val(delta)
