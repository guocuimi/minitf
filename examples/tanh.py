import matplotlib.pyplot as plt

import minitf as tf


def tanh(x):
    y = tf.exp(-x)
    return (1.0 - y) / (1.0 + y)


x = tf.linspace(-7, 7, 200)
with tf.GradientTape() as tp:
    tanh_val = tanh(x)
    grad_tanh = tp.gradient(tanh_val, x)
    grad2_tanh = tp.gradient(grad_tanh, x)
    grad3_tanh = tp.gradient(grad2_tanh, x)
    grad4_tanh = tp.gradient(grad3_tanh, x)
    grad5_tanh = tp.gradient(grad4_tanh, x)
    grad6_tanh = tp.gradient(grad5_tanh, x)

plt.plot(x.numpy(), tanh_val.numpy(),
         x.numpy(), grad_tanh.numpy(),      # first derivative
         x.numpy(), grad2_tanh.numpy(),     # second derivative
         x.numpy(), grad3_tanh.numpy(),     # third derivative
         x.numpy(), grad4_tanh.numpy(),     # fourth derivative
         x.numpy(), grad5_tanh.numpy(),     # fifth derivative
         x.numpy(), grad6_tanh.numpy(),     # sixth derivative
         )

plt.axis('off')
plt.savefig("tanh.png")
plt.show()
