import random

import interfaces
import tensorflow as tf
import numpy as np
import tf_helpers as th
import rmax_learner
from embedding_mmc_replay_memory import MMCPathTracker
from embedding_mmc_replay_memory import ReplayMemory

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
    embedding, weights = construct_embedding_network(abs_state1, abs_state2, 200, 200,
                                                     512 * num_actions + 1)  # plus 1 for shared bias
    w = tf.reshape(weights[:, 0:512 * num_actions], [-1, 512, num_actions])
    b = tf.reshape(weights[:, 512 * num_actions:512 * num_actions + 1], [-1, 1])
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

def construct_dqn_with_subgoal_embedding(input, abs_state1, abs_state2, frame_history, num_actions):
    input = tf.image.convert_image_dtype(input, tf.float32)
    with tf.variable_scope('a1'):
        a1 = th.fully_connected(tf.concat(1, [abs_state1, abs_state2]), 50, tf.nn.relu)
    with tf.variable_scope('c1'):
        c1 = th.down_convolution(input, 8, 4, frame_history, 32, tf.nn.relu)
    with tf.variable_scope('c2'):
        c2 = th.down_convolution(c1, 4, 2, 32, 64, tf.nn.relu)
    with tf.variable_scope('c3'):
        c3 = th.down_convolution(c2, 3, 1, 64, 64, tf.nn.relu)
        N = np.prod([x.value for x in c3.get_shape()[1:]])
        c3 = tf.reshape(c3, [-1, N])
        ac3 = tf.concat(1, [a1, c3])
    with tf.variable_scope('fc1'):
        fc1 = th.fully_connected(ac3, 512, tf.nn.relu)
    with tf.variable_scope('fc2'):
        q_values = th.fully_connected_shared_bias(fc1, num_actions, lambda x: x)
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

    def __init__(self, abs_size, num_actions, num_abstract_states, gamma=0.99, learning_rate=0.00025, replay_start_size=50000,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_steps=1000000,
                 update_freq=4, target_copy_freq=30000, replay_memory_size=1000000,
                 frame_history=4, batch_size=32, error_clip=1, restore_network_file=None, double=True,
                 use_mmc=True, max_mmc_path_length=1000, mmc_beta=0.2):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.inp_actions = tf.placeholder(tf.float32, [None, num_actions])
        self.max_mmc_path_length = max_mmc_path_length
        self.mmc_beta = mmc_beta
        inp_shape = [None, 84, 84, frame_history]
        inp_dtype = 'uint8'
        assert type(inp_dtype) is str
        self.inp_frames = tf.placeholder(inp_dtype, inp_shape)
        self.inp_sp_frames = tf.placeholder(inp_dtype, inp_shape)
        self.inp_terminated = tf.placeholder(tf.bool, [None])
        self.inp_reward = tf.placeholder(tf.float32, [None])
        self.inp_mmc_reward = tf.placeholder(tf.float32, [None])
        self.inp_mask = tf.placeholder(inp_dtype, [None, frame_history])
        self.inp_sp_mask = tf.placeholder(inp_dtype, [None, frame_history])
        #self.inp_q_choices = tf.placeholder(tf.int32, [None])

        self.inp_abs_state_init = tf.placeholder(tf.float32, [None, abs_size])
        self.inp_abs_state_goal = tf.placeholder(tf.float32, [None, abs_size])
        self.abs_neighbors = dict()
        self.gamma = gamma
        q_constructor = construct_dqn_with_subgoal_embedding
        with tf.variable_scope('online'):
            mask_shape = [-1, 1, 1, frame_history]
            mask = tf.reshape(self.inp_mask, mask_shape)
            masked_input = self.inp_frames * mask
            self.q_online = q_constructor(masked_input, self.inp_abs_state_init, self.inp_abs_state_goal, frame_history, num_actions)
        with tf.variable_scope('target'):
            mask_shape = [-1, 1, 1, frame_history]
            sp_mask = tf.reshape(self.inp_sp_mask, mask_shape)
            masked_sp_input = self.inp_sp_frames * sp_mask
            self.q_target = q_constructor(masked_sp_input, self.inp_abs_state_init, self.inp_abs_state_goal, frame_history, num_actions)

        if double:
            with tf.variable_scope('online', reuse=True):
                self.q_online_prime = q_constructor(masked_sp_input, self.inp_abs_state_init, self.inp_abs_state_goal, frame_history, num_actions)
                print self.q_online_prime
            self.maxQ = tf.gather_nd(self.q_target, tf.transpose(
                [tf.range(0, 32, dtype=tf.int32), tf.cast(tf.argmax(self.q_online_prime, axis=1), tf.int32)], [1, 0]))
        else:
            self.maxQ = tf.reduce_max(self.q_target, reduction_indices=1)

        self.r = tf.sign(self.inp_reward)
        use_backup = tf.cast(tf.logical_not(self.inp_terminated), dtype=tf.float32)
        self.y = self.r + use_backup * gamma * self.maxQ

        self.delta_dqn = tf.reduce_sum(self.inp_actions * self.q_online, reduction_indices=1) - self.y
        self.error_dqn = tf.select(tf.abs(self.delta_dqn) < error_clip, 0.5 * tf.square(self.delta_dqn),
                               error_clip * tf.abs(self.delta_dqn))
        if use_mmc:
            self.delta_mmc = (self.inp_mmc_reward - self.y)
            self.error_mmc = tf.select(tf.abs(self.delta_mmc) < error_clip, 0.5 * tf.square(self.delta_mmc),
                               error_clip * tf.abs(self.delta_mmc))
            # self.delta = (1. - self.mmc_beta) * self.delta_dqn + self.mmc_beta * self.delta_mmc
            self.loss = (1. - self.mmc_beta) * tf.reduce_sum(self.error_dqn) + self.mmc_beta * tf.reduce_sum(self.error_mmc)
        else:
            self.loss = tf.reduce_sum(self.error_dqn)
        self.g = tf.gradients(self.loss, self.q_online)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, centered=True, epsilon=0.01)
        self.train_op = optimizer.minimize(self.loss, var_list=th.get_vars('online'))
        self.copy_op = th.make_copy_op('online', 'target')
        self.saver = tf.train.Saver(var_list=th.get_vars('online'))

        self.use_mmc = use_mmc
        self.replay_buffer = ReplayMemory((84, 84), abs_size, 'uint8', replay_memory_size, frame_history)
        if self.use_mmc:
            self.mmc_tracker = MMCPathTracker(self.replay_buffer, self.max_mmc_path_length, self.gamma)
        self.frame_history = frame_history
        self.replay_start_size = replay_start_size
        self.epsilon = [epsilon_start] * num_abstract_states * num_abstract_states
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

        ####################
        ## Keeping track of progress of actions

        self.samples_per_option = 50
        self.state_samples_for_option = dict()
        self.option_action_ticker = dict()
        self.progress_sample_frequency = 1000

        ####################

    def update_q_values(self):
        S1, Sigma1, Sigma2, SigmaGoal, A, R, MMC_R, S2, T, M1, M2 = self.replay_buffer.sample(self.batch_size)
        Aonehot = np.zeros((self.batch_size, self.num_actions), dtype=np.float32)
        Aonehot[range(len(A)), A] = 1

        # # get random sample of neighbors for each Sigma1
        # SigmaGoal = []
        # R = []
        # l0_terminated = []
        # for (terminal, sigma1, sigma2) in zip(T, Sigma1, Sigma2):
        #     sigma_goal = random.sample(self.abs_neighbors[tuple(sigma1)], 1)[0]
        #     SigmaGoal.append(sigma_goal)
        #
        #     l1_transitioned = tuple(sigma1) != tuple(sigma2)
        #     l0_terminated.append(l1_transitioned or terminal)
        #
        #     if terminal:
        #         r = -1
        #     elif l1_transitioned:
        #         if tuple(sigma2) == tuple(sigma_goal):
        #             r = 1
        #         else:
        #             r = -1
        #     else:
        #         r = 0
        #     R.append(r)

        [_, loss, q_online, maxQ, q_target, r, y, delta_dqn, g] = self.sess.run(
            [self.train_op, self.loss, self.q_online, self.maxQ, self.q_target, self.r, self.y, self.delta_dqn,
             self.g],
            feed_dict={self.inp_frames: S1, self.inp_actions: Aonehot,
                       self.inp_abs_state_init: Sigma1,
                       self.inp_abs_state_goal: SigmaGoal,
                       self.inp_sp_frames: S2, self.inp_reward: R, self.inp_mmc_reward: MMC_R,
                       self.inp_terminated: T, self.inp_mask: M1, self.inp_sp_mask: M2})
        return loss

    def run_learning_episode(self, environment, initial_l1_state_vec, goal_l1_state_vec, initial_l1_state, goal_l1_state, abs_func, abs_vec_func, epsilon, max_episode_steps=100000):
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

        option_key = (key_init, tuple(goal_l1_state_vec),)
        if not self.state_samples_for_option.has_key(option_key):
            self.state_samples_for_option[option_key] = []
            self.option_action_ticker[option_key] = 0

        for steps in range(max_episode_steps):
            if environment.is_current_state_terminal():
                break

            state = environment.get_current_state()

            # Save state sample for this option
            self.option_action_ticker[option_key] += 1
            if len(self.state_samples_for_option[option_key]) < self.samples_per_option:
                self.state_samples_for_option[option_key].append(state)
            elif self.replay_buffer.size() > self.replay_start_size \
                    and self.option_action_ticker[option_key] % self.progress_sample_frequency == 0:
                self.record_option_progress(option_key, initial_l1_state_vec, goal_l1_state_vec, initial_l1_state,
                                            goal_l1_state)

            if np.random.uniform(0, 1) < epsilon:
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

            if initial_l1_state != new_l1_state or is_terminal:
                reward = 1 if new_l1_state == goal_l1_state else -1
                episode_finished = True
            else:
                reward = 0

            if self.use_mmc:
                sars = (state[-1], initial_l1_state_vec, abs_vec_func(new_l1_state),
                        goal_l1_state_vec, action, reward, next_state[-1], is_terminal or episode_finished)
                self.mmc_tracker.append(*sars)
                if is_terminal or episode_finished:
                    self.mmc_tracker.flush()
            else:
                sars = (state[-1], initial_l1_state_vec, abs_vec_func(new_l1_state),
                        goal_l1_state_vec, action, reward, 0, next_state[-1], is_terminal or episode_finished)
                self.replay_buffer.append(*sars)

            if (self.replay_buffer.size() > self.replay_start_size) and (self.action_ticker % self.update_freq == 0):
                loss = self.update_q_values()
            if (self.action_ticker - self.replay_start_size) % self.target_copy_freq == 0:
                self.sess.run(self.copy_op)
            self.action_ticker += 1
            episode_steps += 1

            if episode_finished:
                break

        return episode_steps, total_reward, new_l1_state

    def record_option_progress(self, option_key, initial_l1_state_vec, goal_l1_state_vec, initial_l1_state, goal_l1_state):
        state_input = np.transpose(self.state_samples_for_option[option_key], [0, 2, 3, 1])

        [q_values] = self.sess.run([self.q_online],
                                   feed_dict={self.inp_frames: state_input,
                                              self.inp_mask: np.ones((len(state_input), self.frame_history), dtype=np.float32),
                                              self.inp_abs_state_init: np.repeat([initial_l1_state_vec], len(state_input), axis=0),
                                              self.inp_abs_state_goal: np.repeat([initial_l1_state_vec], len(state_input), axis=0)})

        a = rmax_learner.L1Action(initial_l1_state, goal_l1_state, initial_l1_state_vec, goal_l1_state_vec)
        q_vals_filename = 'action_progress/%s_qs.txt' % str(a)
        # loss_filename = 'action_progress/%s_loss.txt' % str(a)
        with open(q_vals_filename, 'a') as f:
            f.write(str(np.average(q_values)) + '\n')
        # with open(loss_filename, 'a') as f:
        #     f.write(loss)

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