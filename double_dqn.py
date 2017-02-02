import tensorflow as tf
import numpy as np
import interfaces
import network_helpers as nh

from replay_memory import ReplayMemory
import dqn


def fully_connected_shared_bias(inp, neurons, rectifier):
    with tf.variable_scope('full_conv_vars'):
        w = tf.get_variable('w', [inp.get_shape()[1].value, neurons], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))
    fc = rectifier(tf.matmul(inp, w) + tf.tile(b, [neurons]))
    return fc


def hook_double_dqn(inp, num_actions):
    with tf.variable_scope('c1'):
        c1 = dqn.down_convolution(inp, 8, 4, 4, 32, tf.nn.relu)
    with tf.variable_scope('c2'):
        c2 = dqn.down_convolution(c1, 4, 2, 32, 64, tf.nn.relu)
    with tf.variable_scope('c3'):
        c3 = dqn.down_convolution(c2, 3, 1, 64, 64, tf.nn.relu)
        N = np.prod([x.value for x in c3.get_shape()[1:]])
        c3 = tf.reshape(c3, [-1, N])
    with tf.variable_scope('fc1'):
        fc1 = dqn.fully_connected(c3, 512, tf.nn.relu)
    with tf.variable_scope('fc2'):
        q_values = fully_connected_shared_bias(fc1, num_actions, lambda x: x)
    return q_values


class DoubleDQNAgent(dqn.DQNAgent):
    def __init__(self, num_actions, gamma=0.99, learning_rate=0.00025, frame_size=84, replay_start_size=50000,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_steps=1000000,
                 update_freq=4, target_copy_freq=30000, replay_memory_size=1000000,
                 frame_history=4, batch_size=32):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.inp_actions = tf.placeholder(tf.float32, [None, num_actions])
        self.inp_frames = tf.placeholder(tf.uint8, [None, frame_size, frame_size, 4])
        self.inp_sp_frames = tf.placeholder(tf.uint8, [None, frame_size, frame_size, 4])
        self.inp_terminated = tf.placeholder(tf.bool, [None])
        self.inp_reward = tf.placeholder(tf.float32, [None])
        self.inp_mask = tf.placeholder(tf.float32, [None, 4])
        self.inp_sp_mask = tf.placeholder(tf.float32, [None, 4])
        self.gamma = gamma
        with tf.variable_scope('online'):
            float_frames = tf.image.convert_image_dtype(self.inp_frames, tf.float32)
            masked_frames = float_frames * tf.tile(tf.reshape(self.inp_mask, [-1, 1, 1, 4]), [1, frame_size, frame_size, 1])
            self.q_online = hook_double_dqn(masked_frames, num_actions)
        with tf.variable_scope('online', reuse=True):
            float_frames = tf.image.convert_image_dtype(self.inp_sp_frames, tf.float32)
            masked_frames = float_frames * tf.tile(tf.reshape(self.inp_sp_mask, [-1, 1, 1, 4]), [1, frame_size, frame_size, 1])
            self.q_online_prime = hook_double_dqn(masked_frames, num_actions)
        with tf.variable_scope('target'):
            float_sp_frames = tf.image.convert_image_dtype(self.inp_sp_frames, tf.float32)
            masked_sp_frames = float_sp_frames * tf.tile(tf.reshape(self.inp_sp_mask, [-1, 1, 1, 4]), [1, frame_size, frame_size, 1])
            self.q_target = hook_double_dqn(masked_sp_frames, num_actions)

        self.maxQ = tf.gather_nd(self.q_target, tf.transpose([tf.range(0, 32, dtype=tf.int32), tf.cast(tf.argmax(self.q_online_prime, axis=1), tf.int32)], [1, 0]))
        self.r = tf.sign(self.inp_reward)
        use_backup = tf.cast(tf.logical_not(self.inp_terminated), dtype=tf.float32)
        self.y = self.r + use_backup * gamma * self.maxQ
        self.error = tf.clip_by_value(tf.reduce_sum(self.inp_actions * self.q_online, reduction_indices=1) - self.y, -1.0, 1.0)
        self.loss = 0.5 * tf.reduce_sum(tf.square(self.error))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, centered=True)
        self.train_op = optimizer.minimize(self.loss, var_list=nh.get_vars('online'))
        self.copy_op = dqn.make_copy_op('online', 'target')
        self.saver = tf.train.Saver(var_list=nh.get_vars('online'))

        self.replay_buffer = ReplayMemory(replay_memory_size, frame_history)
        self.frame_history = frame_history
        self.replay_start_size = replay_start_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_steps = epsilon_steps
        self.epsilon_delta = (self.epsilon - self.epsilon_min)/self.epsilon_steps
        self.update_freq = update_freq
        self.target_copy_freq = target_copy_freq
        self.action_ticker = 1

        self.num_actions = num_actions
        self.batch_size = batch_size

        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.copy_op)
