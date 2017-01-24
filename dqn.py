import tensorflow as tf
import numpy as np
import interfaces
import network_helpers as nh
from collections import deque

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
    def __init__(self, num_actions, gamma=0.99, learning_rate=0.00005, frame_size=84):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.inp_actions = tf.placeholder(tf.float32, [None, num_actions])
        self.inp_frames = tf.image.convert_image_dtype(tf.placeholder(tf.uint8, [None, 84, 84, 4]), tf.float32)
        self.inp_sp_frames = tf.image.convert_image_dtype(tf.placeholder(tf.uint8, [None, 84, 84, 4]), tf.float32)
        self.inp_terminated = tf.placeholder(tf.float32, [None])
        self.inp_reward = tf.sign(tf.placeholder(tf.float32, [None]))
        self.gamma = gamma
        with tf.variable_scope('network'):
            self.q_network = hook_dqn(self.inp_frames, num_actions)
        with tf.variable_scope('target'):
            self.q_target = hook_dqn(self.inp_sp_frames, num_actions)
        self.copy_op = make_copy_op('network', 'target')
        maxQ = tf.reduce_max(self.q_target, reduction_indices=1)
        y = self.inp_reward + (1 - self.inp_terminated) * gamma * maxQ
        self.loss = tf.reduce_mean(tf.square(tf.reduce_sum(self.inp_actions * self.q_network, reduction_indices=1) - y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, epsilon=0.1)
        gradients = optimizer.compute_gradients(self.loss, var_list=nh.get_vars('network'))
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        self.train_op = optimizer.apply_gradients(capped_gvs)
        self.replay_buffer = ReplayBuffer(1000000, 4)
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_steps = 1000000
        self.epsilon_delta = (self.epsilon - self.epsilon_min)/self.epsilon_steps
        self.action_ticker = 1

        self.num_actions = num_actions
        self.batch_size = 32

        self.sess.run(tf.initialize_all_variables())

    def update_q_values(self):
        S1, A, R, S2, T = self.replay_buffer.sample(self.batch_size)
        Aonehot = np.zeros((self.batch_size, self.num_actions))
        Aonehot[range(len(A)), A] = 1

        [_, loss] = self.sess.run([self.train_op, self.loss], feed_dict={self.inp_frames: S1, self.inp_actions: Aonehot,
                                                             self.inp_sp_frames: S2, self.inp_reward: R,
                                                             self.inp_terminated: T})
        return loss

    # TODO add in support for NO-OP frames at the beginning of each episode.
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
            if self.replay_buffer.size() > 50000 and self.action_ticker % 4 == 0:
                loss = self.update_q_values()
            if self.action_ticker % (4*10000) == 0:
                self.sess.run(self.copy_op)
            self.action_ticker += 1
            episode_steps += 1
        return episode_steps, total_reward

    def get_action(self, state):
        state_input = np.transpose(state, [1, 2, 0])
        [q_values] = self.sess.run([self.q_network], feed_dict={self.inp_frames: [state_input]})
        return np.argmax(q_values[0])



class ReplayBuffer(object):

    def __init__(self, capacity, frame_history):
        self.t = 0
        self.filled = False
        self.capacity = capacity
        self.frame_history = frame_history
        # S1 A R S2
        # to grab SARSA(0) -> S(0) A(0) R(0) S(1) T(0)
        self.screens = np.zeros((capacity, 84, 84), dtype=np.uint8)
        self.action = np.zeros(capacity, dtype=np.uint8)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.uint8)

    def append(self, S1, A, R, S2, T):
        self.screens[self.t, :, :] = S1
        self.action[self.t] = A
        self.reward[self.t] = R
        self.terminated[self.t] = T
        self.t = (self.t + 1)
        if self.t >= self.capacity:
            self.t = 0
            self.filled = True

    def get_window(self, array, start, end):
        # these cases aren't exclusive if this isn't true.
        # assert self.capacity > self.frame_history + 1
        if start < 0:
            return np.concatenate((array[start:], array[:end]), axis=0)
        elif end > self.capacity:
            return np.concatenate((array[start:], array[:end-self.capacity]), axis=0)
        else:
            return array[start:end]

    def get_sample(self, index):
        start_frames = index - (self.frame_history - 1)
        end_frames = index + 2
        frames = self.get_window(self.screens, start_frames, end_frames)
        terminations = self.get_window(self.terminated, start_frames, end_frames-1)
        # zeros the frames that are not in the current episode.
        mask = (np.cumsum(terminations[::-1]) == 0)[::-1]

        S0 = np.transpose(frames[:-1], [1, 2, 0]) * mask
        S1 = np.transpose(frames[1:], [1, 2, 0]) * np.concatenate([mask[1:], [1]])
        return S0, self.action[index], self.reward[index], S1, self.terminated[index]

    def sample(self, num_samples):
        if not self.filled:
            idx = np.random.randint(0, self.t, size=num_samples)
        else:
            idx = np.random.randint(0, self.capacity - (self.frame_history + 1), size=num_samples)
            idx = idx - (self.t + self.frame_history + 1)
            idx = idx % self.capacity

        S0 = []
        A = []
        R = []
        S1 = []
        T = []

        for i, sample_i in enumerate(idx):
            s0, a, r, s1, t = self.get_sample(sample_i)
            S0.append(s0)
            A.append(a)
            R.append(r)
            S1.append(s1)
            T.append(t)

        return S0, A, R, S1, T

    def size(self):
        if self.filled:
            return self.capacity
        else:
            return self.t
