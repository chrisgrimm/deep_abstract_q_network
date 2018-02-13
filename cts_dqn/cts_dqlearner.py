import tensorflow as tf
import numpy as np

import tf_helpers as th
from abstract_dqlearner import DQLearner
import toy_mr_encoder
from replay_memory_pc import MMCPathTracker

import cpp_cts
import atari_encoder
from replay_memory import ReplayMemory


class CTSDQLearner(DQLearner, object):
    def __init__(self, config, environment, restore_network_file=None):
        super(CTSDQLearner, self).__init__(config, environment, restore_network_file=restore_network_file)

        # Set configuration params
        self.replay_memory_size = int(self.config['DQL']['REPLAY_MEMORY_SIZE'])
        self.epsilon = float(self.config['DQL']['EPSILON_START'])
        self.epsilon_min = float(self.config['DQL']['EPSILON_END'])
        self.epsilon_steps = int(self.config['DQL']['EPSILON_STEPS'])
        self.epsilon_delta = (self.epsilon - self.epsilon_min) / float(self.epsilon_steps)
        self.cts_size = eval(self.config['CTS']['SIZE'])
        self.bonus_beta = float(self.config['CTS']['BETA'])

        # Setup replay memory
        self.replay_buffer = ReplayMemory(self.frame_shape,
                                          self.frame_dtype,
                                          self.replay_memory_size,
                                          self.frame_history_length)
        if self.use_mmc:
            self.mmc_tracker = MMCPathTracker(self.replay_buffer, self.max_mmc_path_length, self.gamma)

        # Setup CTS
        self.cts = cpp_cts.CPP_CTS(*self.cts_size)
        self.encoding_func = self.get_encoding_func()
        if self.encoding_func is None:
            raise Exception('Encoding function ' + self.config['CTS']['ENCODING_FUNC'] + ' not found')

    def construct_q_network(self, network_input):
        input = tf.image.convert_image_dtype(network_input, tf.float32)
        with tf.variable_scope('c1'):
            c1 = th.down_convolution(input, 8, 4, self.frame_history_length, 32, tf.nn.relu)
        with tf.variable_scope('c2'):
            c2 = th.down_convolution(c1, 4, 2, 32, 64, tf.nn.relu)
        with tf.variable_scope('c3'):
            c3 = th.down_convolution(c2, 3, 1, 64, 64, tf.nn.relu)
            N = np.prod([x.value for x in c3.get_shape()[1:]])
            c3 = tf.reshape(c3, [-1, N])
        with tf.variable_scope('fc1'):
            fc1 = th.fully_connected(c3, 512, tf.nn.relu)
        with tf.variable_scope('fc2'):
            if self.config['DQL']['SHARED_BIAS']:
                q_values = th.fully_connected_shared_bias(fc1, self.num_actions, lambda x: x)
            else:
                q_values = th.fully_connected(fc1, self.num_actions, lambda x: x)
        return q_values

    def update_q_values(self, step, episode_dict):
        S1, A, R, MMC_R, S2, T, M1, M2 = self.replay_buffer.sample(self.batch_size)
        Aonehot = np.zeros((self.batch_size, self.num_actions), dtype=np.float32)
        Aonehot[range(len(A)), A] = 1

        [_, loss] = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={self.inp_frames: S1, self.inp_actions: Aonehot,
                       self.inp_sp_frames: S2, self.inp_reward: R, self.inp_mmc_reward: MMC_R,
                       self.inp_terminated: T, self.inp_mask: M1, self.inp_sp_mask: M2})
        return loss

    def get_action(self, state, environment, episode_dict):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(environment.get_actions_for_state(state))
        else:
            size = list(np.array(range(len(self.frame_shape))) + 1)
            state_input = np.transpose(state, size + [0])
            [q_values] = self.sess.run([self.q_online],
                                       feed_dict={self.inp_frames: [state_input],
                                                  self.inp_mask: np.ones((1, self.frame_history_length), dtype=np.float32)})
            action = np.argmax(q_values[0])

        if self.replay_buffer.size() > self.replay_start_size:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_delta)

        return action

    def record_experience(self, state, action, env_reward, next_state, is_terminal, episode_dict):
        episode_dict['total_reward'] += env_reward

        enc_s = self.encoding_func(self.environment)
        n_hat = self.cts.psuedo_count_for_image(enc_s)
        R_plus = np.sign(env_reward) + (1 - is_terminal) * (self.bonus_beta * np.power(n_hat + 0.01, -0.5))
        R_plus = 1 if R_plus > 1 else R_plus

        if self.use_mmc:
            sars = (state[-1], action, R_plus, next_state[-1], is_terminal)
            self.mmc_tracker.append(*sars)
            if is_terminal:
                self.mmc_tracker.flush()
        else:
            sars = (state[-1], action, R_plus, 0, next_state[-1], is_terminal)
            self.replay_buffer.append(*sars)

    def get_encoding_func(self):
        encoding_func_id = self.config['CTS']['ENCODING_FUNC']

        encoding_func = None

        if encoding_func_id == 'atari':
            encoding_func = atari_encoder.encode_state
        elif encoding_func_id == 'toy_mr':
            encoding_func = toy_mr_encoder.encode_state

        return encoding_func
