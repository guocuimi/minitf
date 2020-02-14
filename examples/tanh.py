import matplotlib.pyplot as plt

import minitf as tf


def tanh(x):
    y = tf.exp(-x)
    return (1.0 - y) / (1.0 + y)


x = tf.Tensor(tf.linspace(-7, 7, 200))
with tf.GradientTape() as tp:
    tanh_val = tanh(x)
    grad_tanh = tp.gradient(tanh_val, x)

# TODO: support high order derivative

plt.plot(x.numpy(), tanh_val.numpy(),
         x.numpy(), grad_tanh.numpy(),  # first derivative
         )

plt.axis('off')
plt.savefig("tanh.png")
plt.show()
