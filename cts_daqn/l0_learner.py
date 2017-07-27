import random

import interfaces
import tensorflow as tf
import numpy as np
import tf_helpers as th
import replay_memory_pc as replay_memory
from cts import cpp_cts

from cts import pc_cts

from cts import toy_mr_encoder


def construct_root_network(input, frame_history):
    input = tf.image.convert_image_dtype(input, tf.float32)
    with tf.variable_scope('c1'):
        c1 = th.down_convolution(input, 8, 4, frame_history, 32, tf.nn.relu)
    with tf.variable_scope('c2'):
        c2 = th.down_convolution(c1, 4, 2, 32, 64, tf.nn.relu)
    with tf.variable_scope('c3'):
        c3 = th.down_convolution(c2, 3, 1, 64, 64, tf.nn.relu)
        N = np.prod([x.value for x in c3.get_shape()[1:]])
        c3 = tf.reshape(c3, [-1, N])
    return c3

def construct_heads_network(input, num_actions, num_abstract_states):
    num_heads = num_abstract_states * num_abstract_states
    with tf.variable_scope('fc1'):
        fc1 = th.fully_connected(input, 512, tf.nn.relu)
    with tf.variable_scope('fc2'):
        q_values = th.fully_connected_multi_shared_bias(fc1, num_actions, num_heads, lambda x: x)
        q_values = tf.reshape(q_values, [-1, num_heads, num_actions])
    return q_values

def construct_dqn_with_embedding(input, abs_state1, abs_state2, frame_history, num_actions):
    embedding, weights = construct_embedding_network(abs_state1, abs_state2, 50, 50, 512*num_actions + 1) # plus 1 for shared bias
    w = tf.reshape(weights[:, 0:512*num_actions], [-1, 512, num_actions])
    b = tf.reshape(weights[:, 512*num_actions:512*num_actions+1], [-1, 1])
    input = tf.image.convert_image_dtype(input, tf.float32)
    with tf.variable_scope('c1'):
        c1 = th.down_convolution(input, 8, 4, frame_history, 32, tf.nn.relu)
    with tf.variable_scope('c2'):
        c2 = th.down_convolution(c1, 4, 2, 32, 64, tf.nn.relu)
    with tf.variable_scope('c3'):
        c3 = th.down_convolution(c2, 3, 1, 64, 64, tf.nn.relu)
        N = np.prod([x.value for x in c3.get_shape()[1:]])
        c3 = tf.reshape(c3, [-1, N])
    with tf.variable_scope('fc1'):
        fc1 = th.fully_connected(c3, 512, tf.nn.relu)
    with tf.variable_scope('fc2'):
        q_values = tf.reshape(tf.matmul(tf.reshape(fc1, [-1, 1, 512]), w), [-1, num_actions]) + b
    return q_values

def construct_dqn_with_embedding_2_layer(input, abs_state1, abs_state2, frame_history, num_actions):
    #embedding, weights = construct_embedding_network(abs_state1, abs_state2, 200, 200,
    #                                                 512 * num_actions + 1)  # plus 1 for shared bias
    #w = tf.reshape(weights[:, 0:512 * num_actions], [-1, 512, num_actions])
    #b = tf.reshape(weights[:, 512 * num_actions:512 * num_actions + 1], [-1, 1])
    with tf.variable_scope('moop'):
        w = tf.get_variable('w', shape=[512, num_actions], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[1], initializer=tf.constant_initializer(0))
    input = tf.image.convert_image_dtype(input, tf.float32)
    with tf.variable_scope('c1'):
        c1 = th.down_convolution(input, 8, 4, frame_history, 32, tf.nn.relu)
    with tf.variable_scope('c2'):
        c2 = th.down_convolution(c1, 4, 2, 32, 64, tf.nn.relu)
    with tf.variable_scope('c3'):
        c3 = th.down_convolution(c2, 3, 1, 64, 64, tf.nn.relu)
        N = np.prod([x.value for x in c3.get_shape()[1:]])
        c3 = tf.reshape(c3, [-1, N])
    with tf.variable_scope('fc1'):
        fc1 = th.fully_connected(c3, 512, tf.nn.relu)
    with tf.variable_scope('fc2'):
        #q_values = tf.reshape(tf.matmul(tf.reshape(fc1, [-1, 1, 512]), w), [-1, num_actions]) + b
        q_values = tf.matmul(fc1, w) + b
    return q_values


def construct_embedding_network(abs_state1, abs_state2, hidden_size, embedding_size, weight_size):
    def shared_abs(inp, neurons):
        with tf.variable_scope('fc1'):
            fc1 = th.fully_connected(inp, neurons, tf.nn.relu)
        with tf.variable_scope('fc2'):
            A = th.fully_connected(fc1, neurons, tf.nn.relu)
        return A
    with tf.variable_scope('A'):
        A1 = shared_abs(abs_state1, hidden_size)
    with tf.variable_scope('A', reuse=True):
        A2 = shared_abs(abs_state2, hidden_size)
    with tf.variable_scope('pre_embedding'):
        pre_embedding = th.fully_connected(tf.concat(1, [A1, A2]), hidden_size*2, tf.nn.relu)
    with tf.variable_scope('embedding'):
        embedding = th.fully_connected(pre_embedding, embedding_size, lambda x: x)
    with tf.variable_scope('pre_weights'):
        pre_weights = th.fully_connected(embedding, embedding_size, tf.nn.relu)
    with tf.variable_scope('weights'):
        weights = th.fully_connected(pre_weights, weight_size, lambda x: x)
    return embedding, weights





class MultiHeadedDQLearner():

    def __init__(self, abs_size, num_actions, num_abstract_states, gamma=0.99, learning_rate=0.000002, replay_start_size=500,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_steps=1000000,
                 update_freq=4, target_copy_freq=30000, replay_memory_size=1000000,
                 frame_history=4, batch_size=32, error_clip=1, restore_network_file=None, double=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.inp_actions = tf.placeholder(tf.float32, [None, num_actions])
        inp_shape = [None, 84, 84, frame_history]
        inp_dtype = 'uint8'
        assert type(inp_dtype) is str
        self.inp_frames = tf.placeholder(inp_dtype, inp_shape)
        self.inp_sp_frames = tf.placeholder(inp_dtype, inp_shape)
        self.inp_terminated = tf.placeholder(tf.bool, [None])
        self.inp_reward = tf.placeholder(tf.float32, [None])
        self.inp_mask = tf.placeholder(inp_dtype, [None, frame_history])
        self.inp_sp_mask = tf.placeholder(inp_dtype, [None, frame_history])
        #self.inp_q_choices = tf.placeholder(tf.int32, [None])

        self.inp_abs_state_init = tf.placeholder(tf.float32, [None, abs_size])
        self.inp_abs_state_goal = tf.placeholder(tf.float32, [None, abs_size])
        self.abs_neighbors = dict()
        self.gamma = gamma

        with tf.variable_scope('online'):
            mask_shape = [-1, 1, 1, frame_history]
            mask = tf.reshape(self.inp_mask, mask_shape)
            masked_input = self.inp_frames * mask
            self.q_online = construct_dqn_with_embedding_2_layer(masked_input, self.inp_abs_state_init, self.inp_abs_state_goal, frame_history, num_actions)
        with tf.variable_scope('target'):
            mask_shape = [-1, 1, 1, frame_history]
            sp_mask = tf.reshape(self.inp_sp_mask, mask_shape)
            masked_sp_input = self.inp_sp_frames * sp_mask
            self.q_target = construct_dqn_with_embedding_2_layer(masked_sp_input, self.inp_abs_state_init, self.inp_abs_state_goal, frame_history, num_actions)

        if double:
            with tf.variable_scope('online', reuse=True):
                self.q_online_prime = construct_dqn_with_embedding_2_layer(masked_sp_input, self.inp_abs_state_init, self.inp_abs_state_goal, frame_history, num_actions)
                print self.q_online_prime
            self.maxQ = tf.gather_nd(self.q_target, tf.transpose(
                [tf.range(0, 32, dtype=tf.int32), tf.cast(tf.argmax(self.q_online_prime, axis=1), tf.int32)], [1, 0]))
        else:
            self.maxQ = tf.reduce_max(self.q_target, reduction_indices=1)

        self.r = self.inp_reward
        use_backup = tf.cast(tf.logical_not(self.inp_terminated), dtype=tf.float32)
        self.y = self.r + use_backup * gamma * self.maxQ
        self.delta = tf.reduce_sum(self.inp_actions * self.q_online, axis=1) - self.y
        self.error = tf.where(tf.abs(self.delta) < error_clip, 0.5 * tf.square(self.delta),
                               error_clip * tf.abs(self.delta))
        self.loss = tf.reduce_sum(self.error)
        self.g = tf.gradients(self.loss, self.q_online)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, centered=True, epsilon=0.01)
        self.train_op = optimizer.minimize(self.loss, var_list=th.get_vars('online'))
        self.copy_op = th.make_copy_op('online', 'target')
        self.saver = tf.train.Saver(var_list=th.get_vars('online'))

        self.replay_buffer = replay_memory.ReplayMemory((84, 84), abs_size, 'uint8', replay_memory_size, frame_history)
        self.frame_history = frame_history
        self.replay_start_size = replay_start_size
        self.epsilon = dict()
        self.epsilon_min = epsilon_end
        self.epsilon_steps = epsilon_steps
        self.epsilon_delta = (epsilon_start - self.epsilon_min) / self.epsilon_steps
        self.update_freq = update_freq
        self.target_copy_freq = target_copy_freq
        self.action_ticker = 1

        self.num_actions = num_actions
        self.batch_size = batch_size

        self.sess.run(tf.initialize_all_variables())

        if restore_network_file is not None:
            self.saver.restore(self.sess, restore_network_file)
            print 'Restored network from file'
        self.sess.run(self.copy_op)

        self.cts = dict()
        self.encoding_func = toy_mr_encoder.encode_toy_mr_state
        self.beta = 0.05

    def update_q_values(self):
        S1, Sigma1, Sigma2, A, R_plus, S2, T, enc_s, M1, M2 = self.replay_buffer.sample(self.batch_size)
        Aonehot = np.zeros((self.batch_size, self.num_actions), dtype=np.float32)
        Aonehot[range(len(A)), A] = 1

        # get random sample of neighbors for each Sigma1
        SigmaGoal = []
        R = []
        for (terminal, sigma1, sigma2, r_plus) in zip(T, Sigma1, Sigma2, R_plus):
            # TODO: SIGMA2 IS NOT SIGMA_GOAL!!!
            sigma_goal = random.sample(self.abs_neighbors[tuple(sigma1)], 1)[0]
            SigmaGoal.append(sigma_goal)

            # check if explore
            if tuple(sigma1) == sigma_goal:
                # augment reward of explore actions
                r = r_plus
            elif terminal:
                r = -1
            elif tuple(sigma1) != tuple(sigma2):
                if tuple(sigma2) == sigma_goal:
                    r = 1
                else:
                    r = -1
            else:
                r = 0
            R.append(r)

        [_, loss, q_online, maxQ, q_target, r, y, error, delta, g] = self.sess.run(
            [self.train_op, self.loss, self.q_online, self.maxQ, self.q_target, self.r, self.y, self.error, self.delta,
             self.g],
            feed_dict={self.inp_frames: S1, self.inp_actions: Aonehot,
                       self.inp_abs_state_init: Sigma1,
                       self.inp_abs_state_goal: SigmaGoal,
                       self.inp_sp_frames: S2, self.inp_reward: R,
                       self.inp_terminated: T, self.inp_mask: M1, self.inp_sp_mask: M2})
        return loss

    def run_learning_episode(self, environment, initial_l1_state_vec, goal_l1_state_vec, initial_l1_state, goal_l1_state, abs_func, abs_vec_func, max_episode_steps=100000):
        episode_steps = 0
        total_reward = 0
        episode_finished = False
        new_l1_state = initial_l1_state
        dqn_tuple = (initial_l1_state, goal_l1_state)
        if dqn_tuple not in self.epsilon:
            self.epsilon[dqn_tuple] = 1.0

        key_init = tuple(initial_l1_state_vec)
        if not self.abs_neighbors.has_key(key_init):
            self.abs_neighbors[key_init] = set()
            self.abs_neighbors[key_init].add(key_init)

            self.cts[key_init] = cpp_cts.CPP_CTS(11, 12, 3)

        for steps in range(max_episode_steps):
            if environment.is_current_state_terminal():
                break

            state = environment.get_current_state()
            if np.random.uniform(0, 1) < self.epsilon[dqn_tuple]:
                action = np.random.choice(environment.get_actions_for_state(state))
            else:
                action = self.get_action(state, initial_l1_state_vec, goal_l1_state_vec)

            if self.replay_buffer.size() > self.replay_start_size:
                self.epsilon[dqn_tuple] = max(self.epsilon_min, self.epsilon[dqn_tuple] - self.epsilon_delta)

            state, action, env_reward, next_state, is_terminal = environment.perform_action(action)
            total_reward += env_reward

            new_l1_state = abs_func(state)
            if initial_l1_state != new_l1_state:
                self.abs_neighbors[key_init].add(tuple(abs_vec_func(new_l1_state)))

                episode_finished = True

            # if initial_l1_state != new_l1_state or is_terminal:
            #     reward = 1 if new_l1_state == goal_l1_state else -1
            #     episode_finished = True
            # else:
            #     reward = 0

            enc_s = self.encoding_func(environment)
            # p, p_prime = self.cts[key_init].prob_update(enc_s)
            # n_hat = 0 if p_prime == p else (p * (1 - p_prime))/(p_prime - p)
            n_hat = max(self.cts[key_init].psuedo_count_for_image(enc_s), 0)
            R_plus = (1 - is_terminal) * (self.beta * np.power(n_hat + 0.01, -0.5))

            self.replay_buffer.append(state[-1], initial_l1_state_vec, abs_vec_func(new_l1_state), action, R_plus, next_state[-1], enc_s, episode_finished or is_terminal)
            if (self.replay_buffer.size() > self.replay_start_size) and (self.action_ticker % self.update_freq == 0):
                loss = self.update_q_values()
            if (self.action_ticker - self.replay_start_size) % self.target_copy_freq == 0:
                self.sess.run(self.copy_op)
            self.action_ticker += 1
            episode_steps += 1

            if episode_finished:
                break

        return episode_steps, total_reward, new_l1_state

    def get_action(self, state, initial_l1_state_vec, goal_l1_state_vec):
        state_input = np.transpose(state, [1, 2, 0])

        [q_values] = self.sess.run([self.q_online],
                                   feed_dict={self.inp_frames: [state_input],
                                              self.inp_mask: np.ones((1, self.frame_history), dtype=np.float32),
                                              self.inp_abs_state_init: [initial_l1_state_vec],
                                              self.inp_abs_state_goal: [goal_l1_state_vec]})
        return np.argmax(q_values[0])

    def save_network(self, file_name):
        self.saver.save(self.sess, file_name)