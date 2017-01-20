import tensorflow as tf
import numpy as np
import interfaces
import network_helpers as nh

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
    source_vars = nh.get_vars(source_scope)
    dest_vars = nh.get_vars(dest_scope)
    ops = [tf.assign(dest_var, source_var) for source_var, dest_var in zip(source_vars, dest_vars)]
    return ops

class DQN_Agent(interfaces.LearningAgent):

    def __init__(self, sess, num_actions, gamma=0.99, learning_rate=0.00005, frame_size=84):
        self.sess = sess
        self.inp_actions = tf.placeholder(tf.float32, [None, num_actions])
        self.inp_frames = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.targets = tf.placeholder(tf.float32, [None, num_actions])
        self.inp_terminated = tf.placeholder(tf.float32, [None])
        self.gamma = gamma
        with tf.variable_scope('network'):
            self.q_network = hook_dqn(self.inp_frames, num_actions)
        with tf.variable_scope('target'):
            self.q_target = hook_dqn(self.inp_frames, num_actions)
        self.copy_op = make_copy_op('network', 'target')


        maxQ = tf.reduce_max(self.q_target, reduction_indices=1)
        y = (1 - self.inp_terminated) * gamma * maxQ
        self.loss = tf.reduce_mean(tf.square(tf.reduce_sum(self.inp_actions * self.q_network, reduction_indices=1) - y))
        # TODO FIGURE OUT RMS PROP STUFF! AHHHHH!
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95)
        gradients = optimizer.compute_gradients(self.loss, var_list=nh.get_vars('network'))
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        self.train_op = optimizer.apply_gradients(capped_gvs)




    def run_learning_episode(self, environment):
        pass

    def get_action(self, state):
        pass

