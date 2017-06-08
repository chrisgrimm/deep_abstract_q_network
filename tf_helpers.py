import tensorflow as tf
import numpy as np


def down_convolution(inp, kernel, stride, filter_in, filter_out, rectifier):
    with tf.variable_scope('conv_vars'):
        w = tf.get_variable('w', [kernel, kernel, filter_in, filter_out], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [filter_out], initializer=tf.constant_initializer(0.0))
    c = rectifier(tf.nn.conv2d(inp, w, [1, stride, stride, 1], 'VALID') + b)
    return c


def down_convolution_weights(inp, dqn_numbers, max_dqn_number, kernel, stride, filter_in, filter_out, rectifier):
    batch_size = tf.shape(inp)[0]
    inp = tf.reshape(inp, [batch_size] + [x.value for x in inp.get_shape()[1:]])
    with tf.variable_scope('conv_vars'):
        W = tf.get_variable('w', [max_dqn_number, kernel, kernel, filter_in, filter_out], initializer=tf.contrib.layers.xavier_initializer())
        B = tf.get_variable('b', [max_dqn_number, filter_out], initializer=tf.constant_initializer(0.0))
    w = tf.reshape(tf.gather_nd(W, tf.reshape(dqn_numbers, [-1, 1])), [batch_size, kernel, kernel, filter_in, filter_out])
    b = tf.reshape(tf.gather_nd(B, tf.reshape(dqn_numbers, [-1, 1])), [batch_size, filter_out])
    print inp
    print tf.nn.conv3d(tf.expand_dims(inp, 0), w, [1, 1, stride, stride, 1], 'VALID')
    c = rectifier(tf.nn.conv3d(tf.expand_dims(inp, 0), w, [1, 1, stride, stride, 1], 'VALID')[0] + tf.reshape(b, [batch_size, 1, 1, filter_out]))
    print c
    #c = tf.reshape(c, [batch_size, tf.shape(c)[2], tf.shape(c)[3], tf.shape(c)[4]])
    print c
    return c


def up_convolution(inp, kernel, filter_in, filter_out, rectifier, bias=0.0):
    [h, w, c] = [x.value for x in inp.get_shape()[1:]]
    with tf.variable_scope('deconv_vars'):
        w1 = tf.get_variable('w1', shape=[kernel, kernel, filter_out, filter_in], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b1', shape=[filter_out], initializer=tf.constant_initializer(bias))
    return rectifier(tf.nn.conv2d_transpose(inp, w1, [32, 2*h, 2*w, filter_out], [1, 2, 2, 1]) + b)


def fully_connected(inp, neurons, rectifier, bias=0.0):
    with tf.variable_scope('full_conv_vars'):
        w = tf.get_variable('w', [inp.get_shape()[1].value, neurons], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [neurons], initializer=tf.constant_initializer(bias))
    fc = rectifier(tf.matmul(inp, w) + b)
    return fc

def fully_connected_weights(inp, dqn_numbers, max_dqn_number, neurons, rectifier, bias=0.0):
    batch_size = tf.shape(inp)[0]
    with tf.variable_scope('full_conv_vars'):
        W = tf.get_variable('W', [max_dqn_number, inp.get_shape()[1].value, neurons], initializer=tf.contrib.layers.xavier_initializer())
        B = tf.get_variable('B', [max_dqn_number, neurons], initializer=tf.constant_initializer(bias))
    w = tf.reshape(tf.gather_nd(W, tf.reshape(dqn_numbers, [-1, 1])), [batch_size, inp.get_shape()[1].value, neurons])
    b = tf.reshape(tf.gather_nd(B, tf.reshape(dqn_numbers, [-1, 1])), [batch_size, neurons])
    fc = rectifier(tf.reshape(tf.batch_matmul(tf.reshape(inp, [batch_size, 1, inp.get_shape()[1].value]), w), [batch_size, -1]) + b)
    return fc


def fully_connected_shared_bias(inp, neurons, rectifier):
    with tf.variable_scope('full_conv_vars'):
        w = tf.get_variable('w', [inp.get_shape()[1].value, neurons], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))
    fc = rectifier(tf.matmul(inp, w) + tf.tile(b, [neurons]))
    return fc

def fully_connected_multi_shared_bias(inp, num_actions, num_heads, rectifier):
    with tf.variable_scope('full_conv_vars'):
        w = tf.get_variable('w', [inp.get_shape()[1].value, num_actions*num_heads], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [num_heads, 1], initializer=tf.constant_initializer(0.0))
        b = tf.reshape(tf.tile(b, [1, num_actions]), [num_actions*num_heads])
    fc = rectifier(tf.matmul(inp, w) + b)
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


def get_vars(*scopes):
    all_vars = []
    for scope in scopes:
        all_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/')
    return all_vars
