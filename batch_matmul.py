import tensorflow as tf
import numpy as np


def fully_connected_weights(inp, dqn_numbers, max_dqn_number, neurons, rectifier, bias=0.0):
    batch_size = tf.shape(inp)[0]
    with tf.variable_scope('full_conv_vars'):
        W = tf.get_variable('W', [max_dqn_number, inp.get_shape()[1].value, neurons], initializer=tf.contrib.layers.xavier_initializer())
        B = tf.get_variable('B', [max_dqn_number, neurons], initializer=tf.constant_initializer(bias))
    w = tf.reshape(tf.gather_nd(W, tf.reshape(dqn_numbers, [-1, 1])), [batch_size, inp.get_shape()[1].value, neurons])
    b = tf.reshape(tf.gather_nd(B, tf.reshape(dqn_numbers, [-1, 1])), [batch_size, neurons])
    fc = tf.reshape(tf.matmul(tf.reshape(inp, [batch_size, 1, inp.get_shape()[1].value]), w), [batch_size, -1])
    return fc, W, B


batch_size = 2
size = 5
num_dqn = 5


fake_inp = tf.random_uniform([batch_size, size], 0, 1)
dqn_numbers = [3, 0]
a, W, B = fully_connected_weights(fake_inp, dqn_numbers, num_dqn, 10, lambda x: x)
res_a = tf.concat([tf.matmul(tf.expand_dims(fake_inp[0], 0), W[3]), tf.matmul(tf.expand_dims(fake_inp[1], 0), W[0])], axis=0)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
[real_a, real_res_a] = sess.run([a, res_a])
print real_a == real_res_a
