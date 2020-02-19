from .tensor import get_val, Tensor


class Variable(Tensor):
    def __init__(self, value):
        # TODO: make a copy of value
        super(Variable, self).__init__(get_val(value))

    def assign_sub(self, delta):
        self._value -= get_val(delta)
