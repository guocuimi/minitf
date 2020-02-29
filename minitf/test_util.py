import minitf as tf

# Stolen from hanson-ml2 in extra_autodiff.ipynb
# Compute an approximation of the gradients using the equation: ...
def numeric_gradients(func, vars_list, x_vs, eps=0.0001):
    partial_derivatives = []
    for idx in range(len(vars_list)):
        # (f(x + x_v*eps/2) - f(x - x_v*eps/2)) / eps
        vars_list_plus = vars_list[:]
        vars_list_plus[idx] += x_vs[idx] * eps / 2
        f_vars_plus = func(*vars_list_plus)

        vars_list_minus = vars_list[:]
        vars_list_minus[idx] -= x_vs[idx] * eps / 2
        f_vars_minus = func(*vars_list_minus)

        derivative = (f_vars_plus - f_vars_minus) / eps
        partial_derivatives.append(derivative)
    return partial_derivatives


def gradients(func, vars_list, output_grad=None):
    with tf.GradientTape() as tp:
        val = func(*vars_list)
    return tp.gradient(val, vars_list, output_grad)


def check_gradients(func, vars_list, atol=1e-4, rtol=1e-4):
    base_func_eval = func(*vars_list)
    y_shape = tf.shape(base_func_eval)
    y_v = tf.random.randn(*y_shape)

    # reverse mode: vjp: dy/dx * y_v
    grads = gradients(func, vars_list, y_v)

    # all grads should have same shape as vars
    assert len(vars_list) == len(grads)
    for var, grad in zip(vars_list, grads):
        assert tf.shape(var) == tf.shape(grad)

    x_vs = []
    for var in vars_list:
        x_shape = tf.shape(var)
        x_v = tf.random.randn(*x_shape)
        x_vs.append(x_v)

    # forward mode: jvp: x_v * dy/dx
    numeric_grads = numeric_gradients(func, vars_list, x_vs)
    # should have same length
    assert len(numeric_grads) == len(vars_list)

    for numeric_grad in numeric_grads:
        assert tf.shape(base_func_eval) == tf.shape(numeric_grad)

    # should be close for each other
    for idx in range(len(vars_list)):
        # (x_v * dy/dx) * y_v
        jvp_v = tf.dot(tf.flatten(numeric_grads[idx]), tf.flatten(y_v))
        # x_v * (dy/dx * y_v)
        vjp_v = tf.dot(tf.flatten(x_vs[idx]), tf.flatten(grads[idx]))
        assert tf.allclose(
            jvp_v,
            vjp_v,
            atol=atol,
            rtol=rtol)
