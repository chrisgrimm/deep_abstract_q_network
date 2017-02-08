import tensorflow as tf
import numpy as np


def down_convolution(inp, kernel, stride, filter_in, filter_out, rectifier):
    with tf.variable_scope('conv_vars'):
        w = tf.get_variable('w', [kernel, kernel, filter_in, filter_out], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [filter_out], initializer=tf.constant_initializer(0.0))
    c = rectifier(tf.nn.conv2d(inp, w, [1, stride, stride, 1], 'VALID') + b)
    return c


def fully_connected(inp, neurons, rectifier):
    with tf.variable_scope('full_conv_vars'):
        w = tf.get_variable('w', [inp.get_shape()[1].value, neurons], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [neurons], initializer=tf.constant_initializer(0.0))
    fc = rectifier(tf.matmul(inp, w) + b)
    return fc


def fully_connected_shared_bias(inp, neurons, rectifier):
    with tf.variable_scope('full_conv_vars'):
        w = tf.get_variable('w', [inp.get_shape()[1].value, neurons], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))
    fc = rectifier(tf.matmul(inp, w) + tf.tile(b, [neurons]))
    return fc


def hook_dqn(inp, num_actions):
    with tf.variable_scope('c1'):
        c1 = down_convolution(inp, 8, 4, 4, 32, tf.nn.relu)
    with tf.variable_scope('c2'):
        c2 = down_convolution(c1, 4, 2, 32, 64, tf.nn.relu)
    with tf.variable_scope('c3'):
        c3 = down_convolution(c2, 3, 1, 64, 64, tf.nn.relu)
        N = np.prod([x.value for x in c3.get_shape()[1:]])
        c3 = tf.reshape(c3, [-1, N])
    with tf.variable_scope('fc1'):
        fc1 = fully_connected(c3, 512, tf.nn.relu)
    with tf.variable_scope('fc2'):
        q_values = fully_connected(fc1, num_actions, lambda x: x)
    return q_values


def make_copy_op(source_scope, dest_scope):
    source_vars = get_vars(source_scope)
    dest_vars = get_vars(dest_scope)
    ops = [tf.assign(dest_var, source_var) for source_var, dest_var in zip(source_vars, dest_vars)]
    return ops


def verify_copy_op():
    with tf.variable_scope('online/fc1/full_conv_vars', reuse=True):
        w_online = tf.get_variable('w')
        b_online = tf.get_variable('b')
    with tf.variable_scope('target/fc1/full_conv_vars', reuse=True):
        w_target = tf.get_variable('w')
        b_target = tf.get_variable('b')

    weights_equal = tf.reduce_prod(tf.cast(tf.equal(w_online, w_target), tf.float32))
    bias_equal = tf.reduce_prod(tf.cast(tf.equal(b_online, b_target), tf.float32))
    return weights_equal * bias_equal


def get_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
