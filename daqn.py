import tensorflow as tf
import tf_helpers as th
import numpy as np
import interfaces
from replay_memory import ReplayMemory

def hardmax(x):
    r = tf.random_uniform([32], minval=0, maxval=1)
    probs = tf.nn.softmax(x)
    r_tile = tf.tile(tf.expand_dims(r, -1), [1, 10])
    i = tf.argmax(tf.cast(tf.cumsum(probs, axis=1) > r_tile, tf.int32), axis=1)
    ii = []
    for j in range(32):
        ii.append(tf.expand_dims(tf.cast(tf.sparse_to_dense(i[j], [10], [1]), tf.float32), 0))
    I = tf.concat(0, ii)
    eps =  I - probs
    return probs + tf.stop_gradient(eps)

def hook_visual(inp, frame_history):
    with tf.variable_scope('c1'):
        c1 = th.down_convolution(inp, 8, 4, frame_history, 32, tf.nn.relu)
    with tf.variable_scope('c2'):
        c2 = th.down_convolution(c1, 4, 2, 32, 64, tf.nn.relu)
    with tf.variable_scope('c3'):
        c3 = th.down_convolution(c2, 3, 1, 64, 64, tf.nn.relu)
        N = np.prod([x.value for x in c3.get_shape()[1:]])
        c3 = tf.reshape(c3, [-1, N])
    with tf.variable_scope('fc1'):
        vis = th.fully_connected(c3, 512, tf.nn.relu)
    return vis

def hook_abstraction(vis, num_abstract_states):
    with tf.variable_scope('vis'):
        it_doesnt_matter = th.fully_connected(vis, num_abstract_states, lambda x: x)
    return hardmax(it_doesnt_matter)

# inp : (84 x 84 x 4) tf.float329,570 ft
def hook_l0(vis, num_abstract_actions, num_actions):
    with tf.variable_scope('fc2'):
        q_values = tf.reshape(th.fully_connected(vis, num_actions * num_abstract_actions , lambda x: x), [-1, num_abstract_actions, num_actions])
    return q_values

def hook_l1(inp_abstracted, num_abstract_actions):
    with tf.variable_scope('fc1'):
        q_values = th.fully_connected(inp_abstracted, num_abstract_actions, lambda x: x)
    return q_values


def flat_actions_to_state_pairs(flat_index, num_abstract_states):
    mapping = [(i, j) for i in range(num_abstract_states) for j in range(num_abstract_states) if i != j]
    return mapping[flat_index]

def state_pairs_to_flat_actions(i, j, num_abstract_states):
    mapping = [(i, j) for i in range(num_abstract_states) for j in range(num_abstract_states) if i != j]
    return dict(zip(mapping, range(num_abstract_states)))[(i, j)]

def valid_actions_for_sigma(actions_for_sigma, sigma, num_abstract_states):
    return tf.reduce_sum(actions_for_sigma * tf.reshape(sigma, [-1, num_abstract_states, 1]), reduction_indices=[1])

def sample_valid_l1_action(actions_for_sigma, sigma, num_abstract_states):
    valid_actions = np.sum(actions_for_sigma * np.reshape(sigma, [num_abstract_states, 1]), axis=1)
    sampled_valid_action_index = np.choose(np.where(valid_actions == 1)[0])
    onehot = np.zeros((num_abstract_states,), dtype=np.float32)
    onehot[sampled_valid_action_index] = 1
    return onehot



class L0_Learner:

    def __init__(self, sess, abstraction_scope, visual_scope, num_actions, num_abstract_actions, num_abstract_states,
                 gamma=0.99, learning_rate=0.00025, replay_start_size=50000,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_steps=1000000,
                 update_freq=4, target_copy_freq=30000, replay_memory_size=1000000,
                 frame_history=4, batch_size=32, error_clip=1):
        self.sess = sess
        self.num_abstract_actions = num_abstract_actions
        self.num_abstract_states = num_abstract_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.replay_buffer = ReplayMemory(self.dqn.get_input_shape(), self.dqn.get_input_dtype(), replay_memory_size,
                                          frame_history)
        self.abstraction_scope = abstraction_scope

        self.inp_frames = tf.placeholder(tf.uint8, [None, 84, 84, 4])
        self.inp_sp_frames = tf.placeholder(tf.uint8, [None, 84, 84, 4])
        self.inp_terminated = tf.placeholder(tf.bool, [None])
        self.inp_reward = tf.placeholder(tf.float32, [None])
        self.inp_mask = tf.placeholder(tf.uint8, [None, frame_history])
        self.inp_sp_mask = tf.placeholder(tf.uint8, [None, frame_history])
        self.inp_actions = tf.placeholder(tf.float32, [None, num_actions])
        # onehot vector
        self.inp_sigma = tf.placeholder(tf.uint8, [None, self.num_abstract_states])

        self.reward_matrix = -np.ones((num_abstract_states, num_abstract_states, num_abstract_actions), dtype=np.float32)
        # make self transitions 0
        for i in range(num_abstract_states):
            self.reward_matrix[i, i, :] = 0
        # make goal transitions have reward 1
        for a in range(num_abstract_actions):
            i, j = flat_actions_to_state_pairs(a, num_abstract_states)
            self.reward_matrix[i, j, a] = 1

        self.actions_for_sigma = np.zeros((num_abstract_states, num_abstract_actions), dtype=np.float32)
        for a in range(num_abstract_actions):
            i, j = flat_actions_to_state_pairs(a, num_abstract_states)
            self.actions_for_sigma[i, a] = 1


        with tf.variable_scope(visual_scope, reuse=True):
            # mask stuff here
            mask = tf.reshape(self.inp_mask, [-1, 1, 1, 1])
            masked_input = self.inp_frames * mask
            self.visual_output = hook_visual(masked_input, self.frame_history)

            mask_sp = tf.reshape(self.inp_sp_mask, [-1, 1, 1, 1])
            masked_input_sp = self.inp_sp_frames * mask_sp
            self.visual_output_sp = hook_visual(masked_input_sp, self.frame_history)

        with tf.variable_scope('online'):
            self.q_online = hook_l0(self.visual_output, self.num_abstract_actions, self.num_actions)
        with tf.variable_scope('target'):
            self.q_target = hook_l0(self.visual_output_sp, self.num_abstract_actions, self.num_actions)

        # TODO set up double dqn for later experiments.

        # Q matrix is (num_abstract_actions, num_actions), results in vector with max-q for each abstract action.
        self.maxQ = tf.reduce_max(self.q_target, reduction_indices=2)

        with tf.variable_scope(self.abstraction_scope, reuse=True):
            self.sigma_p = tf.stop_gradient(hook_abstraction(self.visual_output_sp, num_abstract_states))

        self.r = tf.reduce_sum(
            tf.reshape(self.sigma_p, [-1, 1, num_abstract_states, 1]) * \
            tf.reshape(self.inp_sigma, [-1, num_abstract_states, 1, 1]) * \
            tf.reshape(self.reward_matrix, [1, num_abstract_states, num_abstract_states, num_abstract_actions]),
            reduction_indices=[1, 2])

        use_backup = tf.cast(tf.logical_not(self.inp_terminated), dtype=tf.float32) * tf.reduce_sum(self.sigma_p * self.inp_sigma, reduction_indices=1)
        self.y = self.r + tf.reshape(use_backup, [-1, 1]) * gamma * self.maxQ
        self.delta = tf.reduce_sum(tf.reshape(self.inp_actions, [-1, 1, num_actions]) * self.q_online, reduction_indices=2) - self.y
        valid_actions_mask = valid_actions_for_sigma(self.actions_for_sigma, self.sigma_p, self.num_abstract_actions)
        self.masked_delta = self.delta * valid_actions_mask

        self.error = tf.select(tf.abs(self.masked_delta) < error_clip, 0.5 * tf.square(self.masked_delta),
                               error_clip * tf.abs(self.masked_delta))
        self.loss = tf.reduce_sum(self.error)
        self.g = tf.gradients(self.loss, self.q_online)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, centered=True, epsilon=0.01)
        self.train_op = optimizer.minimize(self.loss, var_list=th.get_vars('online'))
        self.copy_op = th.make_copy_op('online', 'target')
        self.saver = tf.train.Saver(var_list=th.get_vars('online'))

        self.replay_buffer = ReplayMemory((84, 84), tf.uint8, replay_memory_size, frame_history)
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

    def run_learning_episode(self, initial_sigma, l1_action, environment):
        R = 0
        episode_steps = 0
        sigma_p = initial_sigma
        while not environment.is_current_state_terminal():
            state = environment.get_current_state()
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(environment.get_actions_for_state(state))
            else:
                action = self.get_action(state, l1_action)

            if self.replay_buffer.size() > self.replay_start_size:
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_delta)

            s, a, r, sp, t = environment.perform_action(action)
            self.replay_buffer.append(s[-1], a, r, sp[-1], t)
            R += r # TODO: discount?

            if (self.replay_buffer.size() > self.replay_start_size) and (self.action_ticker % self.update_freq == 0):
                loss = self.update_q_values()
            if (self.action_ticker - self.replay_start_size) % self.target_copy_freq == 0:
                self.sess.run(self.copy_op)
            self.action_ticker += 1
            episode_steps += 1

            sigma_p = self.get_abstract_state(sp)
            if np.sum(np.abs(initial_sigma - sigma_p)) > 0.1:
                break

        return initial_sigma, l1_action, R, sigma_p, environment.is_current_state_terminal()

    def get_abstract_state(self, l0_state):
        [sigma] = self.sess.run([self.sigma_p], feed_dict={
            self.inp_sp_frames: [l0_state],
            self.inp_sp_mask: np.ones((1, self.frame_history), dtype=np.float32)
        })
        return sigma[0]

    def update_q_values(self):
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


    def get_action(self, state, l1_action):
        [q_values] = self.sess.run([self.q_online],
                                   feed_dict={self.inp_frames: [state],
                                              self.inp_mask: np.ones((1, self.frame_history), dtype=np.float32)})
        q_values_l0_for_l1 = np.sum(q_values[0] * np.reshape(l1_action, [self.num_abstract_actions, 1]), axis=0)
        return np.argmax(q_values_l0_for_l1)


class L1_Learner:
    def __init__(self, num_abstract_states, num_actions, gamma=0.99, learning_rate=0.00025, replay_start_size=50000,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_steps=1000000,
                 update_freq=4, target_copy_freq=30000, replay_memory_size=1000000,
                 frame_history=4, batch_size=32, error_clip=1):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.num_abstract_states = num_abstract_states
        self.sess = tf.Session(config=config)
        self.inp_actions = tf.placeholder(tf.float32, [None, num_actions])
        inp_shape = [None, 84, 84, 4]
        inp_dtype = tf.uint8
        assert type(inp_dtype) is str
        self.inp_frames = tf.placeholder(inp_dtype, inp_shape)
        self.inp_sp_frames = tf.placeholder(inp_dtype, inp_shape)
        self.inp_terminated = tf.placeholder(tf.bool, [None])
        self.inp_reward = tf.placeholder(tf.float32, [None])
        self.inp_mask = tf.placeholder(inp_dtype, [None, frame_history])
        self.inp_sp_mask = tf.placeholder(inp_dtype, [None, frame_history])
        self.gamma = gamma

        self.num_abstract_states = num_abstract_states
        self.num_abstract_actions = num_abstract_states * (num_abstract_states - 1)
        self.actions_for_sigma = np.zeros((self.num_abstract_states, self.num_abstract_actions), dtype=np.float32)
        for a in range(self.num_abstract_actions):
            i, j = flat_actions_to_state_pairs(a, num_abstract_states)
            self.actions_for_sigma[i, a] = 1

        self.visual_scope = 'visual'
        self.abstraction_scope = 'abstraction'
        with tf.variable_scope(self.visual_scope):
            # mask stuff here
            mask = tf.reshape(self.inp_mask, [-1, 1, 1, 1])
            masked_input = self.inp_frames * mask
            self.visual_output = hook_visual(masked_input, self.frame_history)
        with tf.variable_scope(self.abstraction_scope):
            self.sigma = hook_abstraction(self.visual_output, self.num_abstract_states)

        with tf.variable_scope(self.visual_scope, reuse=True):
            mask_sp = tf.reshape(self.inp_sp_mask, [-1, 1, 1, 1])
            masked_input_sp = self.inp_sp_frames * mask_sp
            self.visual_output_sp = hook_visual(masked_input_sp, self.frame_history)
        with tf.variable_scope(self.abstraction_scope, reuse=True):
            self.sigma_p = hook_abstraction(self.visual_output_sp, self.num_abstract_states)


        possible_action_vector = valid_actions_for_sigma(self.actions_for_sigma, self.sigma_p, self.num_abstract_actions)
        with tf.variable_scope('online'):
            self.q_online = hook_l1(self.sigma, self.num_abstract_actions)
        with tf.variable_scope('online', reuse=True):
            possible_action_vector_prime = -np.inf * (1 - valid_actions_for_sigma(self.actions_for_sigma, self.sigma_p, self.num_abstract_actions))
            self.q_target = possible_action_vector_prime + hook_l1(self.sigma_p, self.num_abstract_actions)

        self.maxQ = tf.reduce_max(self.q_target, reduction_indices=1)

        self.r = tf.sign(self.inp_reward)
        use_backup = tf.cast(tf.logical_not(self.inp_terminated), dtype=tf.float32)
        self.y = self.r + use_backup * gamma * self.maxQ
        self.delta = possible_action_vector * tf.reduce_sum(self.inp_actions * self.q_online, reduction_indices=1) - self.y
        self.error = tf.select(tf.abs(self.delta) < error_clip, 0.5 * tf.square(self.delta),
                               error_clip * tf.abs(self.delta))
        self.loss = tf.reduce_sum(self.error)
        self.g = tf.gradients(self.loss, self.q_online)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, centered=True, epsilon=0.01)
        self.train_op = optimizer.minimize(self.loss, var_list=th.get_vars('online'))
        self.copy_op = th.make_copy_op('online', 'target')
        self.saver = tf.train.Saver(var_list=th.get_vars('online'))

        self.replay_buffer = ReplayMemory((self.num_abstract_states,), np.uint8, replay_memory_size, 1)
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

        self.l0_learner = L0_Learner(self.sess, self.abstraction_scope, self.visual_scope, num_actions, self.num_abstract_actions, self.num_abstract_states)

        self.sess.run(tf.initialize_all_variables())

        self.sess.run(self.copy_op)

    def update_q_values(self):
        S1, A, R, S2, T, M1, M2 = self.replay_buffer.sample(self.batch_size)
        Aonehot = np.zeros((self.batch_size, self.num_abstract_actions), dtype=np.float32)
        Aonehot[range(len(A)), A] = 1

        [_, loss, q_online, maxQ, q_target, r, y, error, delta, g] = self.sess.run(
            [self.train_op, self.loss, self.q_online, self.maxQ, self.q_target, self.r, self.y, self.error, self.delta,
             self.g],
            feed_dict={self.inp_frames: S1, self.inp_actions: Aonehot,
                       self.inp_sp_frames: S2, self.inp_reward: R,
                       self.inp_terminated: T, self.inp_mask: M1, self.inp_sp_mask: M2})
        return loss


    def run_learning_episode(self, environment, max_episode_steps=100000):
        episode_steps = 0
        total_reward = 0
        for steps in range(max_episode_steps):
            s = environment.get_current_state()

            sigma, alpha = self.get_action(s)
            if np.random.uniform(0, 1) < self.epsilon:
                alpha = sample_valid_l1_action(self.actions_for_sigma, self.sigma, self.num_abstract_states)

            if self.replay_buffer.size() > self.replay_start_size:
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_delta)

            sigma, alpha, R, sigma_p, t = self.l0_learner.run_learning_episode(sigma, alpha, environment)
            total_reward += R

            self.replay_buffer.append(np.argmax(sigma), np.argmax(alpha), R, np.argmax(sigma_p), t)

            if self.replay_buffer.size() > self.replay_start_size:
                loss = self.update_q_values()

            self.action_ticker += 1
            episode_steps += 1
        return episode_steps, total_reward

    def get_action(self, state):
        state_input = np.transpose(state, [1, 2, 0])

        [sigma, q_values] = self.sess.run([self.q_online],
                                   feed_dict={self.inp_frames: [state_input],
                                              self.inp_mask: np.ones((1, self.frame_history), dtype=np.float32)})
        return sigma, np.argmax(q_values[0])

    def save_network(self, file_name):
        self.saver.save(self.sess, file_name)