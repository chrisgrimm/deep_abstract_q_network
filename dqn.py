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
    # TODO divide frames by 255.0
    def __init__(self, num_actions, gamma=0.99, learning_rate=0.00005, frame_size=84):
        self.sess = tf.Session()
        self.inp_actions = tf.placeholder(tf.float32, [None, num_actions])
        self.inp_frames = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.inp_sp_frames = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.inp_terminated = tf.placeholder(tf.float32, [None])
        self.inp_reward = tf.placeholder(tf.float32, [None])
        self.gamma = gamma
        with tf.variable_scope('network'):
            self.q_network = hook_dqn(self.inp_frames, num_actions)
        with tf.variable_scope('target'):
            self.q_target = hook_dqn(self.inp_sp_frames, num_actions)
        self.copy_op = make_copy_op('network', 'target')
        maxQ = tf.reduce_max(self.q_target, reduction_indices=1)
        y = self.inp_reward + (1 - self.inp_terminated) * gamma * maxQ
        self.loss = tf.reduce_mean(tf.square(tf.reduce_sum(self.inp_actions * self.q_network, reduction_indices=1) - y))
        # TODO FIGURE OUT RMS PROP STUFF! AHHHHH!
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95)
        gradients = optimizer.compute_gradients(self.loss, var_list=nh.get_vars('network'))
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        self.train_op = optimizer.apply_gradients(capped_gvs)
        self.replay_buffer = ReplayBuffer(1000000, 4)
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_steps = 1000000
        self.action_ticker = 1


    def update_q_values(self):
        # TODO convert actions into onehot
        S1, A, R, S2, T  = zip(*self.replay_buffer.sample(32))
        [_, loss] = self.sess.run([self.train_op, self.loss], feed_dict={self.inp_frames: S1, self.inp_actions: A,
                                                             self.inp_sp_frames: S2, self.inp_reward: R,
                                                             self.inp_terminated: T})
        return loss

    def run_learning_episode(self, environment):
        environment.reset_environment()
        while not environment.is_current_state_terminal():
            state = environment.get_current_state()
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(environment.get_actions_for_state(state))
            else:
                action = self.get_action(state)
            state, action, reward, next_state, is_terminal = environment.perform_action(action)
            self.replay_buffer.append((state, action, reward, next_state, is_terminal))
            if self.replay_buffer.size() > 50000 and self.action_ticker % 4 == 0:
                loss = self.update_q_values()
            if self.action_ticker % 10000 == 0:
                self.sess.run(self.copy_op)


    def get_action(self, state):
        [q_values] = self.sess.run([self.q_network], feed_dict={self.inp_frames: state / 255.0})
        return np.argmax(q_values[0])



class ReplayBuffer(object):

    def __init__(self, capacity, frame_history):
        self.t = 0
        self.filled = False
        self.capacity = capacity
        self.frame_history = frame_history
        # S1 A R S2
        # to grab SARSA(0) -> S(0) A(0) R(0) S(1) T(0)
        self.screens = np.zeros((capacity, 84, 84))
        self.action = np.zeros(capacity)
        self.reward = np.zeros(capacity)
        self.terminated = np.zeros(capacity)

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
        assert self.capacity > self.frame_history + 1
        if start < 0:
            return np.concatenate((array[start:0], array[0:end]), axis=0)
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
        S0 = np.transpose(frames[:-1] * mask, [1, 2, 0])
        S1 = np.transpose(frames[1:] * np.concatenate([mask[1:], [1]]), [1, 2, 0])
        return S0, self.action[index], self.reward[index], S1, self.terminated[index]


    def sample(self, num_samples):
        if not self.filled:
            idx = np.random.randint(0, self.t, size=num_samples)
        else:
            idx = np.random.randint(0, self.capacity - (self.frame_history + 1), size=num_samples)
            idx = idx - (self.t + self.frame_history + 1)
            idx = idx % self.capacity
        samples = [self.get_sample(i) for i in idx]
        return samples

    def size(self):
        if self.filled:
            return self.capacity
        else:
            return self.t
