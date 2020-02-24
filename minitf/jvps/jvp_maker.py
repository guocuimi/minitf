__PRIMITIVE_JVPS = {}


def def_jvp(fun, jvp_maker):
    __PRIMITIVE_JVPS[fun] = jvp_maker


def get_jvp_maker(fun):
    return __PRIMITIVE_JVPS.get(fun)
