import interfaces
import tensorflow as tf
import tf_helpers as th
import numpy as np

class AtariDQN(interfaces.DQNInterface):

    def __init__(self, frame_history, num_actions, shared_bias=True):
        self.frame_history = frame_history
        self.num_actions = num_actions
        self.shared_bias = shared_bias

    def get_input_shape(self):
        return [84, 84]

    def get_input_dtype(self):
        return 'uint8'

    def construct_q_network(self, input):
        input = tf.image.convert_image_dtype(input, tf.float32)
        with tf.variable_scope('c1'):
            c1 = th.down_convolution(input, 8, 4, self.frame_history, 32, tf.nn.relu)
        with tf.variable_scope('c2'):
            c2 = th.down_convolution(c1, 4, 2, 32, 64, tf.nn.relu)
        with tf.variable_scope('c3'):
            c3 = th.down_convolution(c2, 3, 1, 64, 64, tf.nn.relu)
            N = np.prod([x.value for x in c3.get_shape()[1:]])
            c3 = tf.reshape(c3, [-1, N])
        with tf.variable_scope('fc1'):
            fc1 = th.fully_connected(c3, 512, tf.nn.relu)
        with tf.variable_scope('fc2'):
            if self.shared_bias:
                q_values = th.fully_connected_shared_bias(fc1, self.num_actions, lambda x: x)
            else:
                q_values = th.fully_connected(fc1, self.num_actions, lambda x: x)
        return q_values
