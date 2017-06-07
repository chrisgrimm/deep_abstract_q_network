import numpy as np

class ReplayMemory(object):

    def __init__(self, input_shape, abs_size, input_dtype, capacity, frame_history):
        self.t = 0
        self.filled = False
        self.capacity = capacity
        self.frame_history = frame_history
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.abstract_action_numerator_table = dict()
        # S1 A R S2
        # to grab SARSA(0) -> S(0) A(0) R(0) S(1) T(0)
        self.screens = np.zeros([capacity] + list(self.input_shape), dtype=input_dtype)
        self.sigma1 = np.zeros([capacity, abs_size], dtype=np.float32)
        self.sigma2 = np.zeros([capacity, abs_size], dtype=np.float32)
        self.sigma_goal = np.zeros([capacity, abs_size], dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.uint8)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool)
        self.transposed_shape = range(1, len(self.input_shape)+1) + [0]

    def append(self, S1, Sigma1, Sigma2, SigmaGoal, A, R, S2, T):
        if self.filled:
            self.abstract_action_numerator_table[(tuple(self.sigma1[self.t, :]), tuple(self.sigma2[self.t, :]))] -= 1
        self.screens[self.t, :, :] = S1
        self.sigma1[self.t, :] = Sigma1
        self.sigma2[self.t, :] = Sigma2
        self.sigma_goal[self.t, :] = SigmaGoal
        if (tuple(Sigma1), tuple(Sigma2)) in self.abstract_action_numerator_table:
            self.abstract_action_numerator_table[(tuple(Sigma1), tuple(Sigma2))] += 1
        else:
            self.abstract_action_numerator_table[(tuple(Sigma1), tuple(Sigma2))] = 1
        self.action[self.t] = A
        self.reward[self.t] = R
        self.terminated[self.t] = T
        self.t = (self.t + 1)
        if self.t >= self.capacity:
            self.t = 0
            self.filled = True

    def abstract_action_proportions(self, sigma1, sigma2):
        size = len(self.sigma1) if self.filled else self.t
        key = (tuple(sigma1), tuple(sigma2))
        if key in self.abstract_action_numerator_table:
            return self.abstract_action_numerator_table[key] / float(size)
        else:
            return 0

    def get_window(self, array, start, end):
        # these cases aren't exclusive if this isn't true.
        # assert self.capacity > self.frame_history + 1
        if start < 0:
            window = np.concatenate((array[start:], array[:end]), axis=0)
        elif end > self.capacity:
            window = np.concatenate((array[start:], array[:end-self.capacity]), axis=0)
        else:
            window = array[start:end]

        window.flags.writeable = False
        return window

    def get_sample(self, index):
        start_frames = index - (self.frame_history - 1)
        end_frames = index + 2
        frames = self.get_window(self.screens, start_frames, end_frames)
        terminations = self.get_window(self.terminated, start_frames, end_frames-1)
        # zeros the frames that are not in the current episode.
        mask = np.ones((self.frame_history,), dtype=np.float32)
        for i in range(self.frame_history - 2, -1, -1):
            if terminations[i] == 1:
                for k in range(i, -1, -1):
                    mask[k] = 0
                break
        mask2 = np.concatenate((mask[1:], [1]))

        S0 = np.transpose(frames[:-1], self.transposed_shape)
        S1 = np.transpose(frames[1:], self.transposed_shape)

        a = self.action[index]
        r = self.reward[index]
        t = self.terminated[index]
        sigma1 = self.sigma1[index]
        sigma2 = self.sigma2[index]
        sigma_goal = self.sigma_goal[index]

        return S0, sigma1, sigma2, sigma_goal, a, r, S1, t, mask, mask2

    def sample(self, num_samples):
        if not self.filled:
            idx = np.random.randint(0, self.t, size=num_samples)
        else:
            idx = np.random.randint(0, self.capacity - (self.frame_history + 1), size=num_samples)
            idx = idx - (self.t + self.frame_history + 1)
            idx = idx % self.capacity

        S0 = []
        Sigma1 = []
        Sigma2 = []
        SigmaGoal = []
        A = []
        R = []
        S1 = []
        T = []
        M1 = []
        M2 = []

        for sample_i in idx:
            s0, sigma1, sigma2, sigma_goal, a, r, s1, t, mask, mask2 = self.get_sample(sample_i)
            S0.append(s0)
            Sigma1.append(sigma1)
            Sigma2.append(sigma2)
            SigmaGoal.append(sigma_goal)
            A.append(a)
            R.append(r)
            S1.append(s1)
            T.append(t)
            M1.append(mask)
            M2.append(mask2)

        return S0, Sigma1, Sigma2, SigmaGoal, A, R, S1, T, M1, M2

    def size(self):
        if self.filled:
            return self.capacity
        else:
            return self.t
