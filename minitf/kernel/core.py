def primitive(f_raw):
    """
    Wraps a funtion so that its gradient (VJP) can be specified and its invocation can be recorded.
    """
    def f_wrapped(*args, **kwargs):
        # get actual values from tensors
        # TODO: wrape value as tensor here.
        from minitf.tensor import get_val, Tensor
        arg_vals = tuple(map(get_val, args))
        ans = Tensor(f_raw(*arg_vals, **kwargs))

        from minitf.autodiff.graph import register_op
        register_op(f_wrapped, ans, *args, **kwargs)
        return ans

    return f_wrapped


def notrace_primitive(f_raw, as_tensor=True):
    """
    Wraps a function so that it takes Tensor as input and returns Tensor.
    """
    def f_wrapped(*args, **kwargs):
        from minitf.tensor import get_val, Tensor
        # get actual values from tensors
        arg_vals = tuple(map(get_val, args))
        ans = f_raw(*arg_vals, **kwargs)
        return Tensor(ans) if as_tensor else ans

    return f_wrapped
