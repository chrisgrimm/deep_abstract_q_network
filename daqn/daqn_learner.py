import tensorflow as tf
import numpy as np

import tf_helpers as th
from abstract_dqlearner import DQLearner
from daqn.abstraction_tools import abstraction_helpers
from oo_replay_memory import ReplayMemory, MMCPathTracker


class DAQNLearner(DQLearner, object):
    def __init__(self, config, environment, restore_network_file=None):
        # Add tensorflow placeholder
        self.inp_dqn_numbers = tf.placeholder(tf.int32, [None])

        super(DAQNLearner, self).__init__(config, environment, restore_network_file=restore_network_file)

        # Set configuration params
        self.replay_memory_size = int(self.config['DQL']['REPLAY_MEMORY_SIZE'])
        self.max_dqn_number = int(self.config['DAQN']['MAX_DQN_NUMBER'])
        self.abs_func, self.pred_func = abstraction_helpers.get_abstraction_function(self.config['DAQN']['ABS_FUNC_ID'], self.environment)

        # Initialize variables
        self.abs_neighbors = dict()

        # Setup replay memory
        self.replay_buffer = ReplayMemory((84, 84), 'uint8', self.replay_memory_size, self.frame_history_length)
        if self.use_mmc:
            self.mmc_tracker = MMCPathTracker(self.replay_buffer, self.max_mmc_path_length, self.gamma)

    def construct_q_network(self, network_input):
        inp = tf.image.convert_image_dtype(network_input, tf.float32)
        with tf.variable_scope('c1'):
            c1 = th.down_convolution(inp, 8, 4, self.frame_history_length, 32, tf.nn.relu)
        with tf.variable_scope('c2'):
            c2 = th.down_convolution(c1, 4, 2, 32, 64, tf.nn.relu)
        with tf.variable_scope('c3'):
            c3 = th.down_convolution(c2, 3, 1, 64, 64, tf.nn.relu)
            N = np.prod([x.value for x in c3.get_shape()[1:]])
            c3 = tf.reshape(c3, [-1, N])
        with tf.variable_scope('fc1'):
            fc1 = th.fully_connected(c3, 512, tf.nn.relu)
        with tf.variable_scope('fc2'):
            q_values = th.fully_connected_weights(fc1, self.inp_dqn_numbers, self.max_dqn_number, self.num_actions, lambda x: x)
        return q_values

    def update_q_values(self, step, episode_dict):
        if 'dqn_distribution' in episode_dict.keys():
            S1, DQNNumbers, A, R, R_explore, MMC_R, MMC_R_explore, S2, T, M1, M2 = self.replay_buffer.sample(self.batch_size)
        else:
            S1, DQNNumbers, A, R, R_explore, MMC_R, MMC_R_explore, S2, T, M1, M2 = self.replay_buffer.sample_from_distribution(self.batch_size, episode_dict['dqn_distribution'])

        Aonehot = np.zeros((self.batch_size, self.num_actions), dtype=np.float32)
        Aonehot[range(len(A)), A] = 1

        if np.logical_or(np.array(DQNNumbers) < 0, np.array(DQNNumbers) >= self.max_dqn_number).any():
            print 'DQN Number outside range'

        [_, loss] = self.sess.run(
            [self.train_op, self.loss, self.q_online, self.q_target],
            feed_dict={self.inp_frames: S1, self.inp_actions: Aonehot,
                       self.inp_sp_frames: S2, self.inp_reward: R, self.inp_mmc_reward: MMC_R,
                       self.inp_terminated: T, self.inp_mask: M1, self.inp_sp_mask: M2,
                       self.inp_dqn_numbers: DQNNumbers})
        return loss

    def get_action(self, state, environment, episode_dict):
        epsilon = episode_dict['epsilon']

        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(environment.get_actions_for_state(state))
        else:
            state_input = np.transpose(state, [1, 2, 0])

            [q_values] = self.sess.run([self.q_online],
                                       feed_dict={self.inp_frames: [state_input],
                                                  self.inp_mask: np.ones((1, self.frame_history_length), dtype=np.float32),
                                                  self.inp_dqn_numbers: [episode_dict['dqn_number']]})
            return np.argmax(q_values[0])

        return action

    def record_experience(self, state, action, env_reward, next_state, is_terminal, episode_dict):
        initial_l1_state = episode_dict['initial_l1_state']
        goal_l1_state = episode_dict['goal_l1_state']
        dqn_number = episode_dict['dqn_number']
        new_l1_state = self.abs_func(next_state)

        if initial_l1_state != new_l1_state or is_terminal:
            reward = 1 if new_l1_state == goal_l1_state else 0
            episode_finished = True
        else:
            reward = 0
            episode_finished = False

        R = reward

        if self.use_mmc:
            sars = (state[-1], dqn_number, action, reward, 0, next_state[-1], episode_finished)
            self.mmc_tracker.append(*sars)
            if episode_finished:
                self.mmc_tracker.flush()
        else:
            sars = (state[-1], dqn_number, action, R, 0, 0, 0, next_state[-1], episode_finished)
            self.replay_buffer.append(*sars)
