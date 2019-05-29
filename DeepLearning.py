import tensorflow as tf

learning_rate = 0.001
batch_size = 32
num_steps = 500

display_step = 50
num_input = 5120
num_classes = 2
dropout = 0.7

X = tf.placeholder(tf.float32, [None, num_input, 15])

Y = tf.placeholder(tf.float32, [None, num_classes])

filter = tf.zeros([64, 15, 15])

keep_prob = tf.placeholder(tf.float32)

def conv1d(x, W, b, strides=1):
    x = tf.nn.conv1d(x, W, strides=2, padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

