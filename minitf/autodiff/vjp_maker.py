__PRIMITIVE_VJP_MAKERS = {}


def def_vjp_maker(fun, vjp_maker):
    __PRIMITIVE_VJP_MAKERS[fun] = vjp_maker


def get_vjp_maker(fun):
    return __PRIMITIVE_VJP_MAKERS.get(fun)
