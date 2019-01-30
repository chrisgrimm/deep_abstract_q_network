import tensorflow as tf
import numpy as np
import tqdm

EPS = 10**-6

'''
Creates a different RND network for each DQN head. (we dont want to share weights because generalization is bad)
'''
class RND_Array(object):

    def __init__(self,
                 num_rnds,
                 num_frames_for_normalization,
                 name='rnd_array',
                 reuse=None):
        frame_history = 1
        self.inp_shape = inp_shape = [84, 84, frame_history]
        self.num_rnds = num_rnds

        self.n = num_frames_for_normalization
        self.last_n_frames = [[] for _ in range(num_rnds)]
        self.last_n_rewards = [[] for _ in range(num_rnds)]
        self.s_mean = [np.zeros([84, 84, frame_history], dtype=np.float32) for _ in range(num_rnds)]
        self.s_std = [np.zeros([84, 84, frame_history], dtype=np.float32) for _ in range(num_rnds)]
        self.r_std = [0 for _ in range(num_rnds)]




        # build everything
        self.inp_s = tf.placeholder(tf.uint8, [None]+inp_shape)
        self.inp_s_mean = tf.placeholder(tf.float32, inp_shape)
        self.inp_s_std = tf.placeholder(tf.float32, inp_shape)


        self.vars = self.build_array(name, reuse=reuse)

        # initialize everything
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.variables_initializer(self.vars))

    # updates the normalizations with a new *single* image (shape=[84,84,1]).
    def update_state_normalizations(self, inp_s, dqn_idx):
        print(inp_s.shape, self.inp_shape)
        assert inp_s.shape == tuple(self.inp_shape)
        print(inp_s.dtype)
        inp_s = inp_s / 255.
        self.last_n_frames[dqn_idx].append(inp_s)
        self.last_n_frames[dqn_idx] = self.last_n_frames[dqn_idx][-self.n:]
        self.s_mean[dqn_idx] = np.mean(self.last_n_frames[dqn_idx], axis=0)
        self.s_std[dqn_idx] = np.std(self.last_n_frames[dqn_idx], axis=0)

    def update_reward_normalizations(self, r, dqn_idx):
        self.last_n_rewards[dqn_idx].append(r)
        self.last_n_rewards[dqn_idx] = self.last_n_rewards[dqn_idx][-self.n:]
        self.r_std[dqn_idx] = np.std(self.last_n_rewards[dqn_idx])


    ### functions to interface with.

    # pass in a batch of input states and the DQN index for them, get out a batch of intrinsic rewards
    def get_intrinsic_rewards(self, inp_s, dqn_index):
        #assert len(self.last_n_rewards) == self.n
        if len(self.last_n_rewards[dqn_index]) < self.n:
            return np.random.uniform(0, 1)
        sq_diff = self.rnd_square_diffs[dqn_index]
        reward = self.sess.run([sq_diff], feed_dict={self.inp_s: inp_s,
                                                     self.inp_s_mean: self.s_mean[dqn_index],
                                                     self.inp_s_std: self.s_std})
        return reward / self.r_std[dqn_index]

    # pass in a batch of input states and the DQN index, run a training step for that DQN.
    def train_step(self, inp_s, dqn_index):
        assert len(self.last_n_frames) == self.n
        loss, train_op = self.rnd_losses[dqn_index], self.rnd_train_ops[dqn_index]
        _, _loss = self.sess.run([train_op, loss], feed_dict={self.inp_s: inp_s,
                                                              self.inp_s_mean: self.s_mean[dqn_index],
                                                              self.inp_s_std: self.s_std[dqn_index]})
        return _loss


    ### functions that build up the RND_ARRAY

    def build_array(self, name, reuse=None):
        self.rnd_square_diffs = []
        self.rnd_losses = []
        self.rnd_train_ops = []
        with tf.variable_scope(name, reuse=reuse) as scope:
            print('initializing rnd array...')
            for i in tqdm.tqdm(range(self.num_rnds)):
                square_diffs, loss, train_op = self.build_training_pair('rnd_%s' % i)
                self.rnd_square_diffs.append(square_diffs)
                self.rnd_losses.append(loss)
                self.rnd_train_ops.append(train_op)
            variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.original_name_scope)
        return variables



    def build_training_pair(self, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            f_pred, f_pred_vars = self.build_network(self.inp_s, 'f_pred')
            f_random, f_random_vars = self.build_network(self.inp_s, 'f_random')
            opt = self.build_optimizer()
            square_diffs = tf.reduce_mean(tf.square(f_pred - f_random), axis=1)
            loss = tf.reduce_mean(tf.square(f_pred - f_random))
            train_op = opt.minimize(loss, var_list=f_pred_vars)
            return square_diffs, loss, train_op

    def normalize_input(self, input):
        input = input - tf.reshape(self.inp_s_mean, [1]+self.inp_shape) # subtract mean
        input = input / tf.reshape(self.inp_s_std + EPS, [1]+self.inp_shape) # divide by std
        input = tf.clip_by_value(input, -5, 5) # clip between -5 and 5
        return input

    def build_network(self, inp, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            input = tf.image.convert_image_dtype(inp, tf.float32)
            input = self.normalize_input(input)
            c1 = tf.layers.conv2d(input, 32, 8, 4, 'VALID', activation=tf.nn.relu, name='c1')
            c2 = tf.layers.conv2d(c1, 64, 4, 2, 'VALID', activation=tf.nn.relu, name='c2')
            c3 = tf.layers.conv2d(c2, 64, 3, 1, 'VALID', activation=tf.nn.relu, name='c3')
            N = np.prod([x.value for x in c3.get_shape()[1:]])
            c3 = tf.reshape(c3, [-1, N])
            fc1 = tf.layers.dense(c3, 512, activation=tf.nn.relu, name='fc1')
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.original_name_scope)
            return fc1, vars

    def build_optimizer(self):
        lr = 0.0001
        return tf.train.AdamOptimizer(learning_rate=lr)
