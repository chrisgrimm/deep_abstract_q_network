import configparser

import tensorflow as tf

import interfaces
import tf_helpers as th


class DQLearner(interfaces.LearningAgent):

    def __init__(self, config, environment, restore_network_file=None):

        # Set configuration params
        self.config = config
        self.frame_history_length = int(self.config['ENV']['FRAME_HISTORY_LENGTH'])
        self.frame_shape = eval(self.config['DQL']['FRAME_SHAPE'])
        self.inp_shape = [None] + list(self.frame_shape) + [self.frame_history_length]
        self.frame_dtype = str(self.config['DQL']['FRAME_DTYPE'])
        self.replay_start_size = int(self.config['DQL']['REPLAY_START_SIZE'])
        self.update_freq = int(self.config['DQL']['NETWORK_UPDATE_FREQ'])
        self.target_copy_freq = int(self.config['DQL']['TARGET_COPY_FREQ'])
        self.batch_size = int(self.config['DQL']['BATCH_SIZE'])
        self.max_mmc_path_length = int(self.config['DQL']['MAX_MMC_PATH_LENGTH'])
        self.mmc_beta = float(self.config['DQL']['MMC_BETA'])
        self.gamma = float(self.config['DQL']['GAMMA'])
        self.double = self.config['DQL']['DOUBLE'] == 'True'
        self.use_mmc = self.config['DQL']['USE_MMC'] == 'True'
        error_clip = float(self.config['DQL']['ERROR_CLIP'])
        learning_rate = float(self.config['DQL']['LEARNING_RATE'])

        self.environment = environment
        self.num_actions = len(self.environment.get_actions_for_state(None))
        self.action_ticker = 1

        # Set tensorflow config
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True
        self.sess = tf.Session(config=tf_config)

        # Setup tensorflow placeholders
        assert type(self.frame_dtype) is str
        self.inp_actions = tf.placeholder(tf.float32, [None, self.num_actions], name='actions')
        self.inp_frames = tf.placeholder(self.frame_dtype, self.inp_shape, name='frames')
        self.inp_sp_frames = tf.placeholder(self.frame_dtype, self.inp_shape, name='sp_frames')
        self.inp_terminated = tf.placeholder(tf.bool, [None], name='terminated')
        self.inp_reward = tf.placeholder(tf.float32, [None], name='reward')
        self.inp_mmc_reward = tf.placeholder(tf.float32, [None], name='mmc_reward')
        self.inp_mask = tf.placeholder(self.frame_dtype, [None, self.frame_history_length], name='mask')
        self.inp_sp_mask = tf.placeholder(self.frame_dtype, [None, self.frame_history_length], name='sp_mask')

        # Setup Q-Networks
        with tf.variable_scope('online'):
            mask_shape = [-1, 1, 1, self.frame_history_length]
            mask = tf.reshape(self.inp_mask, mask_shape)
            masked_input = self.inp_frames * mask
            self.q_online = self.construct_q_network(masked_input)
        with tf.variable_scope('target'):
            mask_shape = [-1, 1, 1, self.frame_history_length]
            sp_mask = tf.reshape(self.inp_sp_mask, mask_shape)
            masked_sp_input = self.inp_sp_frames * sp_mask
            self.q_target = self.construct_q_network(masked_sp_input)
        if self.double:
            with tf.variable_scope('online', reuse=True):
                self.q_online_prime = self.construct_q_network(masked_sp_input)
                print self.q_online_prime
            self.maxQ = tf.gather_nd(self.q_target, tf.transpose(
                [tf.range(0, self.batch_size, dtype=tf.int32),
                 tf.cast(tf.argmax(self.q_online_prime, axis=1), tf.int32)],
                [1, 0]))
        else:
            self.maxQ = tf.reduce_max(self.q_target, axis=1)

        # Create loss handle
        self.r = tf.sign(self.inp_reward)
        use_backup = tf.cast(tf.logical_not(self.inp_terminated), dtype=tf.float32)
        self.y = self.r + use_backup * self.gamma * self.maxQ
        self.delta_dqn = tf.reduce_sum(self.inp_actions * self.q_online, axis=1) - self.y
        self.error_dqn = tf.where(tf.abs(self.delta_dqn) < error_clip, 0.5 * tf.square(self.delta_dqn),
                                  error_clip * tf.abs(self.delta_dqn))
        if self.use_mmc:
            self.delta_mmc = tf.reduce_sum(self.inp_actions * self.q_online, axis=1) - self.inp_mmc_reward
            self.error_mmc = tf.where(tf.abs(self.delta_mmc) < error_clip, 0.5 * tf.square(self.delta_mmc),
                                      error_clip * tf.abs(self.delta_mmc))
            self.loss = (1. - self.mmc_beta) * tf.reduce_sum(self.error_dqn) + self.mmc_beta * tf.reduce_sum(
                self.error_mmc)
        else:
            self.loss = tf.reduce_sum(self.error_dqn)

        # Create optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, centered=True, epsilon=0.01)
        self.train_op = optimizer.minimize(self.loss, var_list=th.get_vars('online'))
        self.copy_op = th.make_copy_op('online', 'target')
        self.saver = tf.train.Saver(var_list=th.get_vars('online'))
        self.sess.run(tf.global_variables_initializer())

        # Optionally load previous weights
        if restore_network_file is not None:
            self.saver.restore(self.sess, restore_network_file)
            print 'Restored network from file'
        self.sess.run(self.copy_op)

    def save_network(self, file_name):
        self.saver.save(self.sess, file_name)

    def run_learning_episode(self, environment, episode_dict, max_episode_steps=100000):
        episode_steps = 0
        total_reward = 0

        for step in range(max_episode_steps):
            if environment.is_current_state_terminal() or self.extra_termination_conditions(step, episode_dict):
                break

            # Get action
            state = environment.get_current_state()
            action = self.get_action(state, episode_dict)

            # Act
            state, action, env_reward, next_state, is_terminal = environment.perform_action(action)
            total_reward += env_reward

            # Record experience
            self.record_experience(state, action, env_reward, next_state, is_terminal, episode_dict)

            # Update network weights
            if (self.action_ticker > self.replay_start_size) and (self.action_ticker % self.update_freq == 0):
                self.update_q_values(step, episode_dict)

            # Copy target network
            if (self.action_ticker - self.replay_start_size) % self.target_copy_freq == 0:
                self.sess.run(self.copy_op)

            self.action_ticker += 1
            episode_steps += 1

        return episode_steps, total_reward

    def construct_q_network(self, network_input):
        raise NotImplemented

    def update_q_values(self, step, episode_dict):
        raise NotImplemented

    def extra_termination_conditions(self, step, episode_dict):
        return False

    def get_action(self, state, episode_dict):
        raise NotImplemented

    def record_experience(self, state, action, env_reward, next_state, is_terminal, episode_dict):
        raise NotImplemented

