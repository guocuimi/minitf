from .graph import get_current_graph
from .tensor import get_val, Tensor, is_tensor

__PRIMITIVE_JAPS = {}


def def_jvp(fun, jvp_maker):
    __PRIMITIVE_JAPS[fun] = jvp_maker


def get_jvp_maker(fun):
    return __PRIMITIVE_JAPS.get(fun)


def primitive(f_raw):
    """
    Wraps a funtion so that its gradient (JVP) can be specified and its invocation can be recorded.
    """
    def f_wrapped(*args, **kwargs):
        # get actual values from tensors
        arg_vals = tuple(map(get_val, args))
        ans = Tensor(f_raw(*arg_vals, **kwargs))

        current_graph = get_current_graph()
        if current_graph:
            # make jvp functions
            jvp_maker = get_jvp_maker(f_wrapped)
            if jvp_maker is None:
                raise Exception("Need to define jvp for the primitive")
            all_jvps = jvp_maker(ans, *arg_vals, **kwargs)

            tensors = []
            jvps = []
            for arg, jvp in zip(args, all_jvps):
                if is_tensor(arg):
                    tensors.append(arg)
                    jvps.append(jvp)

            # register grad func for each tensor
            current_graph.add_edges(ans, tensors, jvps)
        return ans

    return f_wrapped


def notrace_primitive(f_raw, as_tensor=True):
    """
    Wraps a function so that it takes Tensor as input and returns Tensor.
    """

    def f_wrapped(*args, **kwargs):
        # get actual values from tensors
        arg_vals = tuple(map(get_val, args))
        ans = f_raw(*arg_vals, **kwargs)
        return Tensor(ans) if as_tensor else ans

    return f_wrapped
