import tensorflow as tf
import numpy as  np
# input dimensions
i = 10  # number of samples
s = 5   # number of time steps
n = 3   # number of features

# Generate random input data
inputs = np.random.rand(i, s, n).astype(np.float32)
# Define a SimpleRNN layer
simple_rnn=tf.keras.layers.SimpleRNN(i)
print(inputs)
# Pass the inputs through the SimpleRNN layer
output=simple_rnn(inputs)
print(output)







