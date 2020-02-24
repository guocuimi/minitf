from minitf import kernel as K
from minitf.tensor import Tensor
from minitf.tensor import get_val


class Variable(Tensor):
    def __init__(self, value):
        # TODO: make a copy of value
        super(Variable, self).__init__(get_val(value))

    def __str__(self):
        return "tf.Variable(id=%s, shape=%s, numpy=%s)" % (id(self), K.shape(self.numpy()), str(self.numpy()))

    def __repr__(self):
        return "<tf.Variable: id=%s, shape=%s, numpy=%s>" % (id(self), K.shape(self.numpy()), self.numpy())

    def assign_sub(self, delta):
        self._value -= get_val(delta)
