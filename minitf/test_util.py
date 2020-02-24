import minitf as tf


def unbroadcast(target, g):
    while tf.rank(g) > tf.rank(target):
        g = tf.sum(g, axis=0)
    for axis, size in enumerate(tf.shape(target)):
        if size == 1:
            g = tf.sum(g, axis=axis, keepdims=True)
    return g


# Stolen from hanson-ml2 in extra_autodiff.ipynb
# Compute an approximation of the gradients using the equation: ...
def numeric_gradients(func, vars_list, eps=0.0001):
    partial_derivatives = []
    base_func_eval = func(*vars_list)
    for idx in range(len(vars_list)):
        tweaked_vars = vars_list[:]
        tweaked_vars[idx] += eps
        tweaked_func_eval = func(*tweaked_vars)
        derivative = (tweaked_func_eval - base_func_eval) / eps

        # unbroadcast to generate same shape
        derivative = unbroadcast(vars_list[idx], derivative)
        assert tf.shape(derivative) == tf.shape(vars_list[idx])
        partial_derivatives.append(derivative)
    return partial_derivatives


def gradients(func, vars_list):
    with tf.GradientTape() as tp:
        val = func(*vars_list)
    return tp.gradient(val, vars_list)


def check_gradients(func, vars_list, atol=1e-4, rtol=1e-4):
    grads = gradients(func, vars_list)
    numeric_grads = numeric_gradients(func, vars_list)

    # should have same length
    assert len(numeric_grads) == len(grads)
    # should be close for each other
    for numeric_grad, grad in zip(numeric_grads, grads):
        assert tf.allclose(numeric_grad, grad, atol=atol, rtol=rtol)
