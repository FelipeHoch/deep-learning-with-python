import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_samples_per_class = 1000

# Creating synthetical data
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5], 
         [0.5, 1]],
    size=num_samples_per_class
)

# Creating synthetical data
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], 
         [0.5, 1]],
    size=num_samples_per_class
)

# Stacking the data
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

# Creating the corresponding target labels 0s and 1s
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"), 
                     np.ones((num_samples_per_class, 1), dtype="float32")))



# Now, we'll building our regression linear model from scratch.
how_many_inputs_dimensions = 2
how_many_output_dimensions = 1

W = tf.Variable(initial_value=tf.random.uniform(shape=(how_many_inputs_dimensions, how_many_output_dimensions)))

b = tf.Variable(initial_value=tf.zeros(shape=(how_many_output_dimensions,)))

def model(inputs):
    matrix_multplication = tf.matmul(inputs, W)

    return matrix_multplication + b

def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)

    return tf.reduce_mean(per_sample_losses)

learnig_rate = 0.1

def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)

        loss = square_loss(targets, predictions)

    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W,b])

    W.assign_sub(grad_loss_wrt_W * learnig_rate)

    b.assign_sub(grad_loss_wrt_b * learnig_rate)

    return loss

for step in range(40):
    loss = training_step(inputs, targets)

    print(f"Loss at step {step}: {loss:.4f}")

predictions = model(inputs)

x = np.linspace(-1,4, 100)

y = - W[0] / W[1] * x + (0.5 - b) / W[1]

# plt.plot(x, y, "-r")

# plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)

# plt.show()