import minitf as tf

# Stolen from hanson-ml2 in extra_autodiff.ipynb
# Compute an approximation of the gradients using the equation: ...
def numeric_gradients(func, vars_list, eps=0.0001):
    partial_derivatives = []
    for idx in range(len(vars_list)):
        # (f(x + eps/2) - f(x - eps/2)) / eps
        vars_list_plus = vars_list[:]
        vars_list_plus[idx] += eps / 2
        f_vars_plus = func(*vars_list_plus)

        vars_list_minus = vars_list[:]
        vars_list_minus[idx] -= eps / 2
        f_vars_minus = func(*vars_list_minus)

        derivative = (f_vars_plus - f_vars_minus) / eps
        partial_derivatives.append(derivative)
    return partial_derivatives


def gradients(func, vars_list):
    with tf.GradientTape() as tp:
        val = func(*vars_list)
    return tp.gradient(val, vars_list)


def check_gradients(func, vars_list, atol=1e-4, rtol=1e-4):
    # reverse mode
    grads = gradients(func, vars_list)

    # all grads should have same shape as vars
    assert len(vars_list) == len(grads)
    for var, grad in zip(vars_list, grads):
        assert tf.shape(var) == tf.shape(grad)

    # forward mode
    numeric_grads = numeric_gradients(func, vars_list)
    # should have same length
    assert len(numeric_grads) == len(vars_list)

    base_func_eval = func(*vars_list)
    for numeric_grad in numeric_grads:
        assert tf.shape(base_func_eval) == tf.shape(numeric_grad)

    # should be close for each other
    for numeric_grad, grad in zip(numeric_grads, grads):
        assert tf.allclose(
            tf.reduce_sum(numeric_grad),
            tf.reduce_sum(grad),
            atol=atol,
            rtol=rtol)
