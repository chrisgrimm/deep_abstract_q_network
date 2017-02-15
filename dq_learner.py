import interfaces
import tensorflow as tf
import numpy as np
import tf_helpers as th
from replay_memory import ReplayMemory


class DQLearner(interfaces.LearningAgent):

    def __init__(self, dqn, num_actions, gamma=0.99, learning_rate=0.00025, replay_start_size=50000,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_steps=1000000,
                 update_freq=4, target_copy_freq=30000, replay_memory_size=1000000,
                 frame_history=4, batch_size=32, error_clip=1, restore_network_file=None, double=True, offline_buffer=None):
        self.dqn = dqn
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.inp_actions = tf.placeholder(tf.float32, [None, num_actions])
        inp_shape = [None] + list(self.dqn.get_input_shape()) + [frame_history]
        inp_dtype = self.dqn.get_input_dtype()
        assert type(inp_dtype) is str
        self.inp_frames = tf.placeholder(inp_dtype, inp_shape)
        self.inp_sp_frames = tf.placeholder(inp_dtype, inp_shape)
        self.inp_terminated = tf.placeholder(tf.bool, [None])
        self.inp_reward = tf.placeholder(tf.float32, [None])
        self.inp_mask = tf.placeholder(inp_dtype, [None, frame_history])
        self.inp_sp_mask = tf.placeholder(inp_dtype, [None, frame_history])
        self.offline_buffer = offline_buffer
        self.gamma = gamma
        with tf.variable_scope('online'):
            mask_shape = [-1] + [1]*len(self.dqn.get_input_shape()) + [frame_history]
            mask = tf.reshape(self.inp_mask, mask_shape)
            masked_input = self.inp_frames * mask
            self.q_online = self.dqn.construct_q_network(masked_input)
        with tf.variable_scope('target'):
            mask_shape = [-1] + [1] * len(self.dqn.get_input_shape()) + [frame_history]
            sp_mask = tf.reshape(self.inp_sp_mask, mask_shape)
            masked_sp_input = self.inp_sp_frames * sp_mask
            self.q_target = self.dqn.construct_q_network(masked_sp_input)

        if double:
            with tf.variable_scope('online', reuse=True):
                self.q_online_prime = self.dqn.construct_q_network(masked_sp_input)
            self.maxQ = tf.gather_nd(self.q_target, tf.transpose(
                [tf.range(0, 32, dtype=tf.int32), tf.cast(tf.argmax(self.q_online_prime, axis=1), tf.int32)], [1, 0]))
        else:
            self.maxQ = tf.reduce_max(self.q_target, reduction_indices=1)

        self.r = tf.sign(self.inp_reward)
        use_backup = tf.cast(tf.logical_not(self.inp_terminated), dtype=tf.float32)
        self.y = self.r + use_backup * gamma * self.maxQ
        self.delta = tf.reduce_sum(self.inp_actions * self.q_online, reduction_indices=1) - self.y
        self.error = tf.select(tf.abs(self.delta) < error_clip, 0.5 * tf.square(self.delta),
                               error_clip * tf.abs(self.delta))
        self.loss = tf.reduce_sum(self.error)
        self.g = tf.gradients(self.loss, self.q_online)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, centered=True, epsilon=0.01)
        self.train_op = optimizer.minimize(self.loss, var_list=th.get_vars('online'))
        self.copy_op = th.make_copy_op('online', 'target')
        self.saver = tf.train.Saver(var_list=th.get_vars('online'))

        self.replay_buffer = ReplayMemory(self.dqn.get_input_shape(), self.dqn.get_input_dtype(), replay_memory_size, frame_history)
        self.frame_history = frame_history
        self.replay_start_size = replay_start_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_steps = epsilon_steps
        self.epsilon_delta = (self.epsilon - self.epsilon_min) / self.epsilon_steps
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

    def update_q_values(self, offline=False):
        if offline:
            S1, A, R, S2, T, M1, M2 = self.offline_buffer.sample(self.batch_size)
        else:
            S1, A, R, S2, T, M1, M2 = self.replay_buffer.sample(self.batch_size)
        Aonehot = np.zeros((self.batch_size, self.num_actions), dtype=np.float32)
        Aonehot[range(len(A)), A] = 1

        [_, loss, q_online, maxQ, q_target, r, y, error, delta, g] = self.sess.run(
            [self.train_op, self.loss, self.q_online, self.maxQ, self.q_target, self.r, self.y, self.error, self.delta,
             self.g],
            feed_dict={self.inp_frames: S1, self.inp_actions: Aonehot,
                       self.inp_sp_frames: S2, self.inp_reward: R,
                       self.inp_terminated: T, self.inp_mask: M1, self.inp_sp_mask: M2})
        return loss

    def run_learning_episode(self, environment, max_episode_steps=100000, offline=False):
        episode_steps = 0
        total_reward = 0
        for steps in range(max_episode_steps):
            # supporting offline learning.
            if not offline:
                if environment.is_current_state_terminal():
                    break

                state = environment.get_current_state()
                if np.random.uniform(0, 1) < self.epsilon:
                    action = np.random.choice(environment.get_actions_for_state(state))
                else:
                    action = self.get_action(state)
                if self.replay_buffer.size() > self.replay_start_size:
                    self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_delta)
                state, action, reward, next_state, is_terminal = environment.perform_action(action)
                total_reward += reward
                self.replay_buffer.append(state[-1], action, reward, next_state[-1], is_terminal)
                if (self.replay_buffer.size() > self.replay_start_size) and (self.action_ticker % self.update_freq == 0):
                    loss = self.update_q_values()
            else:
                loss = self.update_q_values(offline=True)
            if (self.action_ticker - self.replay_start_size) % self.target_copy_freq == 0:
                self.sess.run(self.copy_op)
            self.action_ticker += 1
            episode_steps += 1
        return episode_steps, total_reward

    def get_action(self, state):
        state_input = np.transpose(state, [1, 2, 0])

        [q_values] = self.sess.run([self.q_online],
                                   feed_dict={self.inp_frames: [state_input],
                                              self.inp_mask: np.ones((1, self.frame_history), dtype=np.float32)})
        return np.argmax(q_values[0])

    def save_network(self, file_name):
        self.saver.save(self.sess, file_name)

