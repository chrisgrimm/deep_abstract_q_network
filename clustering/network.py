import tensorflow as tf
import numpy as np
import tf_helpers as th

def make_embedding_network(state, embedding_size):
    shape = [x.value for x in state.get_shape()[1:]]
    state = tf.image.convert_image_dtype(tf.reshape(state, [-1] + shape + [1]), tf.float32)
    with tf.variable_scope('c1'):
        c1 = th.down_convolution(state, 5, 2, 1, 32, tf.nn.relu)
    with tf.variable_scope('c2'):
        c2 = th.down_convolution(c1, 5, 2, 32, 64, tf.nn.relu)
    with tf.variable_scope('c3'):
        c3 = th.down_convolution(c2, 5, 2, 64, 64, tf.nn.relu)
        N = np.prod([x.value for x in c3.get_shape()[1:]])
    with tf.variable_scope('fc1'):
        fc1 = th.fully_connected(tf.reshape(c3, [-1, N]), 512, tf.nn.relu)
    with tf.variable_scope('fc2'):
        E = th.fully_connected(fc1, embedding_size, lambda x: x)
    return E

inp_state1 = tf.placeholder(tf.uint8, [None, 84, 84])
inp_state2 = tf.placeholder(tf.uint8, [None, 84, 84])
inp_step = tf.placeholder(tf.uint8, [None])
inp_delta_diff = tf.placeholder(tf.float32)
inp_delta_max = tf.placeholder(tf.float32)

with tf.variable_scope('E'):
    e1 = make_embedding_network(inp_state1, 2)
with tf.variable_scope('E', reuse=True):
    e2 = make_embedding_network(inp_state2, 2)

dist_sq = tf.reduce_sum(tf.square(e1 - e2), axis=1)
is_diff = tf.cast(tf.greater(inp_step, 1), tf.float32)
diff_loss = tf.reduce_max([tf.square(inp_delta_diff) - dist_sq, tf.zeros_like(dist_sq), dist_sq - tf.square(inp_delta_max)], reduction_indices=0)
sim_loss = dist_sq
loss = tf.reduce_mean(is_diff * diff_loss + (1 - is_diff) * sim_loss)

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())

