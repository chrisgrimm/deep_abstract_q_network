from tensorflow.examples.tutorials.mnist import input_data
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

def categorical_onehot(n, x):
    r = tf.random_uniform([32], minval=0, maxval=1)
    probs = tf.nn.softmax(x)
    r_tile = tf.tile(tf.expand_dims(r, -1), [1, 10])
    i = tf.argmax(tf.cast(tf.cumsum(probs, axis=1) > r_tile, tf.int32), axis=1)
    ii = []
    for j in range(32):
        ii.append(tf.expand_dims(tf.cast(tf.sparse_to_dense(i[j], [10], [1]), tf.float32), 0))
    I = tf.concat(0, ii)
    #I = tf.cast(tf.sparse_to_dense(i, [32, 10], tf.ones([32])), tf.float32)
    print I
    print probs
    eps =  I - probs
    return probs + tf.stop_gradient(eps), r_tile, i



inp_mnist = tf.placeholder(tf.float32, [32, 28*28])
inp_target = tf.placeholder(tf.float32, [32, 10])

with tf.variable_scope('c1'):
    c1 = down_convolution(tf.reshape(inp_mnist, [-1, 28, 28, 1]), 5, 2, 1, 100, tf.nn.relu)
with tf.variable_scope('c2'):
    c2 = down_convolution(c1, 5, 2, 100, 100, tf.nn.relu)
    N = np.prod([c.value for c in c2.get_shape()[1:]])
    c2 = tf.reshape(c2, [-1, N])
with tf.variable_scope('fc1'):
    logits = fully_connected(c2, 10, lambda x: x)
    categories, r_tile, indices = categorical_onehot(10, logits)
    probs = tf.nn.softmax(logits)


loss = tf.reduce_mean(tf.square(categories - inp_target))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, inp_target))
train_op = tf.train.AdamOptimizer().minimize(loss)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())
mnist = input_data.read_data_sets('./MNIST', one_hot=True)
NUM_TRAIN = 10000
NUM_TEST = 1000
for i in range(NUM_TRAIN):
    inp_mnist_, inp_target_ = mnist.train.next_batch(32)
    [_, loss_, categories_, probs_, r_tile_, indices_] = sess.run([train_op, loss, categories, probs, r_tile, indices], feed_dict={inp_mnist: inp_mnist_, inp_target: inp_target_})
    print i, loss_


grand_mean = 0
for i in range(NUM_TEST):
    inp_mnist_, inp_target_ = mnist.test.next_batch(32)
    [probs_] = sess.run([probs], feed_dict={inp_mnist: inp_mnist_, inp_target: inp_target_})
    acc = np.mean(np.argmax(probs_, axis=1) == np.argmax(inp_target_, axis=1), 0)
    grand_mean += acc
print grand_mean / float(NUM_TEST)