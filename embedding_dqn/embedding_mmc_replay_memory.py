import numpy as np
from collections import deque

class MMCPathTracker(object):

    def __init__(self, replay_memory, max_path_length, gamma):
        self.replay_memory = replay_memory
        self.max_path_length = max_path_length
        self.gamma = gamma
        self.path = deque()
        self.C = 0
        self.path_length_counter = 0

    def push(self, S1, Sigma1, Sigma2, SigmaGoal, A, R, S2, T):
        self.C += self.gamma ** self.path_length_counter * R
        self.path.append((S1, Sigma1, Sigma2, SigmaGoal, A, R, S2, T))

    def pop(self):
        (S1, Sigma1, Sigma2, SigmaGoal, A, R, S2, T) = self.path.popleft()
        self.replay_memory.append(S1, Sigma1, Sigma2, SigmaGoal, A, R, self.C, S2, T)
        self.C = (self.C - R)/self.gamma

    def append(self, S1, Sigma1, Sigma2, SigmaGoal, A, R, S2, T):
        if len(self.path) == self.max_path_length:
            self.pop()
        self.push(S1, Sigma1, Sigma2, SigmaGoal, A, R, S2, T)
        self.path_length_counter = min(self.path_length_counter + 1, self.max_path_length)

    def flush(self):
        for i in xrange(len(self.path)):
            self.pop()
        self.C = 0
        self.path_length_counter = 0


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
        self.mmc_reward = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool)
        self.transposed_shape = range(1, len(self.input_shape)+1) + [0]

    def append(self, S1, Sigma1, Sigma2, SigmaGoal, A, R, MMCR, S2, T):
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
        self.mmc_reward[self.t] = MMCR
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
        mmc_r = self.mmc_reward[index]
        t = self.terminated[index]
        sigma1 = self.sigma1[index]
        sigma2 = self.sigma2[index]
        sigma_goal = self.sigma_goal[index]

        return S0, sigma1, sigma2, sigma_goal, a, r, mmc_r, S1, t, mask, mask2

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
        MMC_R = []
        S1 = []
        T = []
        M1 = []
        M2 = []

        for sample_i in idx:
            s0, sigma1, sigma2, sigma_goal, a, r, mmc_r, s1, t, mask, mask2 = self.get_sample(sample_i)
            S0.append(s0)
            Sigma1.append(sigma1)
            Sigma2.append(sigma2)
            SigmaGoal.append(sigma_goal)
            A.append(a)
            R.append(r)
            MMC_R.append(mmc_r)
            S1.append(s1)
            T.append(t)
            M1.append(mask)
            M2.append(mask2)

        return S0, Sigma1, Sigma2, SigmaGoal, A, R, MMC_R, S1, T, M1, M2

    def size(self):
        if self.filled:
            return self.capacity
        else:
            return self.t
