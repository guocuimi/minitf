import minitf as tf

tf.random.set_seed(42)

# generate some linear-looking data
def synthetic_linear_data(w, b, num_examples):
    X = tf.random.normal(size=num_examples)
    noise = tf.random.normal(size=num_examples)
    Y = X * w + b + noise
    return X, Y


true_w = 3.0
true_b = 2.0
num_examples = 500
X, Y = synthetic_linear_data(true_w, true_b, num_examples)


# linear model
def linear_model(X, W, b):
    return tf.dot(X, W) + b


# loss function
def loss(Y_hat, Y):
    return tf.reduce_mean(tf.square(Y_hat - Y) / 2)


def apply_gradients(grads_and_vars):
    for gradient, var in grads_and_vars:
        var.assign_sub(learning_rate * gradient)


# parameters to train
w = tf.Variable(5.0)
b = tf.Variable(0.0)
parameters = [w, b]

learning_rate = 0.1
# train loop
for epoch in range(10):
    # forward pass
    with tf.GradientTape() as tp:
        Y_hat = linear_model(X, w, b)
        current_loss = loss(Y_hat, Y)

    gradients = tp.gradient(current_loss, parameters)

    # apply gradient
    apply_gradients(zip(gradients, parameters))

    print('Epoch %2d: w=%1.2f b=%1.2f, loss=%2.5f' %
          (epoch, w.numpy(), b.numpy(), current_loss.numpy()))
