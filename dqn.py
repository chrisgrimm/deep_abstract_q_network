import tensorflow as tf
import numpy as np
import interfaces
import network_helpers as nh

from replay_memory import ReplayMemory

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


def verify_copy_op():
    with tf.variable_scope('network/fc1/full_conv_vars', reuse=True):
        w_network = tf.get_variable('w')
        b_network = tf.get_variable('b')
    with tf.variable_scope('target/fc1/full_conv_vars', reuse=True):
        w_target = tf.get_variable('w')
        b_target = tf.get_variable('b')

    weights_equal = tf.reduce_prod(tf.cast(tf.equal(w_network, w_target), tf.float32))
    bias_equal = tf.reduce_prod(tf.cast(tf.equal(b_network, b_target), tf.float32))
    return weights_equal * bias_equal


class DQNAgent(interfaces.LearningAgent):
    def __init__(self, num_actions, gamma=0.99, learning_rate=0.00005, frame_size=84, replay_start_size=50000,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_steps=1000000,
                 update_freq=4, target_copy_freq=10000, replay_memory_size=1000000,
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
        with tf.variable_scope('network'):
            float_frames = tf.image.convert_image_dtype(self.inp_frames, tf.float32)
            masked_frames = float_frames * tf.tile(tf.reshape(self.inp_mask, [-1, 1, 1, 4]), [1, frame_size, frame_size, 1])
            self.q_network = hook_dqn(masked_frames, num_actions)
        with tf.variable_scope('target'):
            float_sp_frames = tf.image.convert_image_dtype(self.inp_sp_frames, tf.float32)
            masked_sp_frames = float_sp_frames * tf.tile(tf.reshape(self.inp_sp_mask, [-1, 1, 1, 4]), [1, frame_size, frame_size, 1])
            self.q_target = hook_dqn(masked_sp_frames, num_actions)

        self.maxQ = tf.reduce_max(self.q_target, reduction_indices=1)
        self.r = tf.sign(self.inp_reward)
        use_backup = tf.cast(tf.logical_not(self.inp_terminated), dtype=tf.float32)
        self.y = self.r + use_backup * gamma * self.maxQ
        self.error = tf.clip_by_value(tf.reduce_sum(self.inp_actions * self.q_network, reduction_indices=1) - self.y, -1.0, 1.0)
        self.loss = tf.reduce_sum(tf.square(self.error))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95)
        self.train_op = optimizer.minimize(self.loss, var_list=nh.get_vars('network'))
        self.copy_op = make_copy_op('network', 'target')
        self.saver = tf.train.Saver(var_list=nh.get_vars('network'))

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

    def update_q_values(self):
        S1, A, R, S2, T, M1, M2 = self.replay_buffer.sample(self.batch_size)
        Aonehot = np.zeros((self.batch_size, self.num_actions), dtype=np.float32)
        Aonehot[range(len(A)), A] = 1

        [_, loss, q_network, maxQ, q_target, r, y, error] = self.sess.run([self.train_op, self.loss, self.q_network, self.maxQ, self.q_target, self.r, self.y, self.error],
                                  feed_dict={self.inp_frames: S1, self.inp_actions: Aonehot,
                                             self.inp_sp_frames: S2, self.inp_reward: R,
                                             self.inp_terminated: T, self.inp_mask: M1, self.inp_sp_mask: M2})
        return loss

    def run_learning_episode(self, environment):
        environment.reset_environment()
        episode_steps = 0
        total_reward = 0
        while not environment.is_current_state_terminal():
            state = environment.get_current_state()
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(environment.get_actions_for_state(state))
            else:
                action = self.get_action(state)
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_delta)

            state, action, reward, next_state, is_terminal = environment.perform_action(action)
            total_reward += reward
            self.replay_buffer.append(state[-1], action, reward, next_state[-1], is_terminal)
            if self.replay_buffer.size() > self.replay_start_size and self.action_ticker % self.update_freq == 0:
                loss = self.update_q_values()
            if (self.action_ticker - self.replay_start_size) % (self.update_freq * self.target_copy_freq) == 0:
                self.sess.run(self.copy_op)
            self.action_ticker += 1
            episode_steps += 1
        return episode_steps, total_reward

    def get_action(self, state):
        state_input = np.transpose(state, [1, 2, 0])
        [q_values] = self.sess.run([self.q_network],
                                   feed_dict={self.inp_frames: [state_input],
                                              self.inp_mask: np.ones((1, self.frame_history), dtype=np.float32)})
        return np.argmax(q_values[0])

    def save_network(self, file_name):
        self.saver.save(self.sess, file_name)
