import tensorflow as tf
import numpy as np

import tf_helpers as th
from abstract_dqlearner import DQLearner
from replay_memory import ReplayMemory


class VanillaDQLearner(DQLearner, object):
    def __init__(self, config, environment, restore_network_file=None):
        super(VanillaDQLearner, self).__init__(config, environment, restore_network_file=restore_network_file)

        # Set configuration params
        self.replay_memory_size = int(self.config['DQL']['REPLAY_MEMORY_SIZE'])
        self.epsilon = float(self.config['DQL']['EPSILON_START'])
        self.epsilon_min = float(self.config['DQL']['EPSILON_END'])
        self.epsilon_steps = int(self.config['DQL']['EPSILON_STEPS'])
        self.epsilon_delta = (self.epsilon - self.epsilon_min) / float(self.epsilon_steps)

        # Setup replay memory
        self.replay_buffer = ReplayMemory(self.frame_shape, self.frame_dtype, self.replay_memory_size, self.frame_history_length)

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
        S1, A, R, S2, T, M1, M2 = self.replay_buffer.sample(self.batch_size)
        Aonehot = np.zeros((self.batch_size, self.num_actions), dtype=np.float32)
        Aonehot[range(len(A)), A] = 1

        [_, loss] = self.sess.run([self.train_op, self.loss],
                                  feed_dict={self.inp_frames: S1, self.inp_actions: Aonehot,
                                             self.inp_sp_frames: S2, self.inp_reward: R,
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
        self.replay_buffer.append(state[-1], action, env_reward, next_state[-1], is_terminal)
