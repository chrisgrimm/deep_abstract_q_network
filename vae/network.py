import tensorflow as tf
import numpy as np
import tf_helpers as th

def leaky_relu(x, alpha=0.001):
    return tf.maximum(alpha * x, x)

def make_encoder(inp, encoding_size):
    with tf.variable_scope('c1'):
        c1 = th.down_convolution(inp, 5, 2, 1, 32, tf.nn.relu)
    with tf.variable_scope('c2'):
        c2 = th.down_convolution(c1, 5, 2, 32, 64, tf.nn.relu)
    with tf.variable_scope('c3'):
        c3 = th.down_convolution(c2, 5, 2, 64, 64, tf.nn.relu)
        N = np.prod([x.value for x in c3.get_shape()[1:]])
        c3 = tf.reshape(c3, [-1, N])
    with tf.variable_scope('mu_zGx'):
        with tf.variable_scope('fc1'):
            fc1 = th.fully_connected(c3, encoding_size, tf.nn.relu)
        with tf.variable_scope('fc2'):
            mu = th.fully_connected(fc1, encoding_size, lambda x:x)
    with tf.variable_scope('sigma_zGx'):
        with tf.variable_scope('fc1'):
            fc1 = th.fully_connected(c3, encoding_size, tf.nn.relu)
        with tf.variable_scope('fc2'):
            sigma = th.fully_connected(fc1, encoding_size, tf.nn.relu)
    return mu, sigma

def make_decoder(z):
    with tf.variable_scope('fc1'):
        fc1 = tf.reshape(th.fully_connected(z, 21*21*64, tf.nn.relu), [-1, 21, 21, 64])
    with tf.variable_scope('d1'):
        d1 = th.up_convolution(fc1, 5, 64, 32, tf.nn.relu)
    with tf.variable_scope('d2_mu'):
        mu_x = th.up_convolution(d1, 5, 32, 1, lambda x: x)
    with tf.variable_scope('d2_sigma'):
        sigma_x = th.up_convolution(d1, 5, 32, 1, tf.nn.relu)

    return mu_x, sigma_x

encoding_size = 50
batch_size = 32
inp_image = tf.placeholder(tf.float32, [None, 84, 84, 1])

with tf.variable_scope('encoder'):
    mu_z, sigma_z = make_encoder(inp_image, encoding_size)
z = sigma_z * tf.random_normal([batch_size, encoding_size]) + mu_z
with tf.variable_scope('decoder'):
    mu_x, sigma_x = make_decoder(z)

term1 = 0.5 * tf.reduce_sum(1 + 2*tf.log(sigma_z + 10**-8) - tf.square(mu_z) - tf.square(sigma_z), reduction_indices=[1])
k = 84*84
term2 = -k / 2.0 * np.log(2*np.pi) - 0.5*tf.reduce_sum(2*tf.log(sigma_x + 10**-8), reduction_indices=[1, 2, 3]) - tf.reduce_sum(0.5*tf.square(inp_image - mu_x)*(1.0 / (tf.square(sigma_x) + 10**-8)), [1, 2, 3])
loss = -tf.reduce_mean((term1 + term2), reduction_indices=0)

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())