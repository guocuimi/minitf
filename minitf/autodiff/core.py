from .graph import get_current_graph
from .tensor import get_val, Tensor, is_tensor

primitive_jvps = {}


def def_jvp(fun, jvp_maker):
    primitive_jvps[fun] = jvp_maker


def get_jvp_maker(fun):
    return primitive_jvps.get(fun)


def primitive(f_raw):
    def f_wrapped(*args, **kwargs):
        # get actual values from tensors
        argvals = tuple(map(get_val, args))
        ans = Tensor(f_raw(*argvals, **kwargs))

        current_graph = get_current_graph()
        if current_graph:
            # make jvp functions
            jvp_maker = get_jvp_maker(f_wrapped)
            if jvp_maker is None:
                raise Exception("Need to define jvp for the primitive")
            all_jvps = jvp_maker(ans, *argvals, **kwargs)

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
