import random

import numpy as np
from collections import deque


class MMCPathTracker(object):

    def __init__(self, replay_memory, max_path_length, gamma):
        self.replay_memory = replay_memory
        self.max_path_length = max_path_length
        self.gamma = gamma
        self.replay_path = deque()
        self.path = np.zeros(shape=[max_path_length], dtype=np.float32)
        self.mmc_reward_array = np.ones(shape=[max_path_length], dtype=np.float32)
        for i in range(max_path_length):
            self.mmc_reward_array[i] = self.gamma ** i
        self.path_start_index = 0
        self.path_end_index = 0

    def _get_path_slice(self):
        if self.path_start_index < self.path_end_index:
            path_slice = self.path[self.path_start_index:self.path_end_index]
        else:
            p1 = self.path[self.path_start_index:]
            p2 = self.path[:self.path_end_index]
            path_slice = np.concatenate([p1, p2])
        path_slice = np.pad(path_slice, (0, len(self.mmc_reward_array) - len(path_slice)), 'constant')
        return path_slice

    def _get_mmc_reward_for_slice(self, slice):
        return np.sum(slice * self.mmc_reward_array)

    def _push(self, S1, DQNNumber, A, R, S2, T):
        self.path[self.path_end_index] = R
        self.replay_path.append((S1, DQNNumber, A, R, S2, T))
        self.path_end_index = (self.path_end_index + 1) % self.max_path_length

    def _pop(self):
        (S1, DQNNumber, A, R, S2, T) = self.replay_path.popleft()
        MMCR = self._get_mmc_reward_for_slice(self._get_path_slice())
        self.replay_memory.append(S1, DQNNumber, A, R, MMCR, S2, T)
        self.path_start_index = (self.path_start_index + 1) % self.max_path_length

    def append(self, S1, DQNNumber, A, R, S2, T):
        if len(self.replay_path) == self.max_path_length:
            self._pop()
        self._push(S1, DQNNumber, A, R, S2, T)

    def flush(self):
        for i in xrange(len(self.replay_path)):
            self._pop()
        self.path.fill(0)
        self.path_start_index = 0
        self.path_end_index = 0


class MMCPathTrackerExplore(object):

    def __init__(self, replay_memory, max_path_length, gamma):
        self.replay_memory = replay_memory
        self.max_path_length = max_path_length
        self.gamma = gamma
        self.replay_path = deque()
        self.path = np.zeros(shape=[max_path_length], dtype=np.float32)
        self.path_explore = np.zeros(shape=[max_path_length], dtype=np.float32)
        self.mmc_reward_array = np.ones(shape=[max_path_length], dtype=np.float32)
        for i in range(max_path_length):
            self.mmc_reward_array[i] = self.gamma ** i
        self.path_start_index = 0
        self.path_end_index = 0

    def _get_path_slice(self, path):
        if self.path_start_index < self.path_end_index:
            path_slice = path[self.path_start_index:self.path_end_index]
        else:
            p1 = path[self.path_start_index:]
            p2 = path[:self.path_end_index]
            path_slice = np.concatenate([p1, p2])
        path_slice = np.pad(path_slice, (0, len(self.mmc_reward_array) - len(path_slice)), 'constant')
        return path_slice

    def _get_mmc_reward_for_slice(self, slice):
        return np.sum(slice * self.mmc_reward_array)

    def _push(self, S1, DQNNumber, A, R, R_explore, S2, T):
        self.path[self.path_end_index] = R
        self.path_explore[self.path_end_index] = R_explore
        self.replay_path.append((S1, DQNNumber, A, R, R_explore, S2, T))
        self.path_end_index = (self.path_end_index + 1) % self.max_path_length

    def _pop(self):
        (S1, DQNNumber, A, R, R_explore, S2, T) = self.replay_path.popleft()
        MMCR = self._get_mmc_reward_for_slice(self._get_path_slice(self.path))
        MMCR_explore = self._get_mmc_reward_for_slice(self._get_path_slice(self.path_explore))
        self.replay_memory.append(S1, DQNNumber, A, R, R_explore, MMCR, MMCR_explore, S2, T)
        self.path_start_index = (self.path_start_index + 1) % self.max_path_length

    def append(self, S1, DQNNumber, A, R, R_explore, S2, T):
        if len(self.replay_path) == self.max_path_length:
            self._pop()
        self._push(S1, DQNNumber, A, R, R_explore, S2, T)

    def flush(self):
        for i in xrange(len(self.replay_path)):
            self._pop()
        self.path.fill(0)
        self.path_start_index = 0
        self.path_end_index = 0


class IndexList(object):
    def __init__(self, max_size):
        self.l = np.zeros(max_size, dtype=np.uint32)
        self.start = 0
        self.end = 0
        self.max_size = max_size
        self.size = 0

    def add(self, i):
        self.l[self.end] = i
        self.end += 1
        self.end %= self.max_size
        self.size += 1
        assert self.size <= self.max_size

    def pop(self, item=None):
        if item is None or (len(self) > 0 and self[0] == item):
            self.start += 1
            self.start %= self.max_size
            self.size -= 1
            assert self.size >= 0

    def __getitem__(self, index):
        if not 0 <= index < self.size: raise IndexError()
        return self.l[(self.start + index) % self.max_size]

    def __len__(self):
        return self.size
        # if self.end < self.start:
        #     return self.end - self.start
        # else:
        #     return self.max_size - self.start + self.end

    def sample(self):
        i = np.random.randint(len(self))
        return self[i]


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
        self.dqn_numbers = np.zeros([capacity], dtype=np.uint32)
        self.action = np.zeros(capacity, dtype=np.uint8)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.mmc_reward = np.zeros(capacity, dtype=np.float32)
        self.reward_explore = np.zeros(capacity, dtype=np.float32)
        self.mmc_reward_explore = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool)
        self.transposed_shape = range(1, len(self.input_shape)+1) + [0]
        self.dqn_indices = dict()

    def append(self, S1, DQNNumber, A, R, R_explore, MMCR, MMCR_explore, S2, T):
        self.screens[self.t, :, :] = S1
        self.dqn_numbers[self.t] = DQNNumber
        self.action[self.t] = A
        self.reward[self.t] = R
        self.mmc_reward[self.t] = MMCR
        self.reward_explore[self.t] = R_explore
        self.mmc_reward_explore[self.t] = MMCR_explore
        self.terminated[self.t] = T
        self.t = (self.t + 1)
        if self.t >= self.capacity:
            self.t = 0
            self.filled = True

        for i in self.dqn_indices:
            self.dqn_indices[i].pop(self.t)
        if DQNNumber not in self.dqn_indices:
            self.dqn_indices[DQNNumber] = IndexList(self.capacity)
        self.dqn_indices[DQNNumber].add(self.t)

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
        r_explore = self.reward_explore[index]
        mmc_r_explore = self.mmc_reward_explore[index]
        t = self.terminated[index]
        dqn_number = self.dqn_numbers[index]

        return S0, dqn_number, a, r, r_explore, mmc_r, mmc_r_explore, S1, t, mask, mask2

    def sample(self, num_samples):
        if not self.filled:
            idx = np.random.randint(0, self.t, size=num_samples)
        else:
            idx = np.random.randint(0, self.capacity - (self.frame_history + 1), size=num_samples)
            idx = idx - (self.t + self.frame_history + 1)
            idx = idx % self.capacity
        return self.collect_from_indices(idx)

    def sample_from_distribution(self, num_samples, distribution):
        for key, value in distribution.items():
            distribution[key] = value*len(self.dqn_indices[key])
        sum = np.sum(distribution.values())
        for key in distribution:
            distribution[key] /= sum

        keys, values = zip(*distribution.items())
        dqn_numbers = np.random.choice(keys, p=values, size=num_samples)
        idx = []
        for n in dqn_numbers:
            idx.append(self.dqn_indices[n].sample())
        return self.collect_from_indices(idx)

    def collect_from_indices(self, idx):
        S0 = []
        DQNNumbers = []
        A = []
        R = []
        MMC_R = []
        R_explore = []
        MMC_R_explore = []
        S1 = []
        T = []
        M1 = []
        M2 = []

        for sample_i in idx:
            s0, dqn_number, a, r, r_explore, mmc_r, mmc_r_explore, s1, t, mask, mask2 = self.get_sample(sample_i)
            S0.append(s0)
            DQNNumbers.append(dqn_number)
            A.append(a)
            R.append(r)
            MMC_R.append(mmc_r)
            R_explore.append(r_explore)
            MMC_R_explore.append(mmc_r_explore)
            S1.append(s1)
            T.append(t)
            M1.append(mask)
            M2.append(mask2)

        return S0, DQNNumbers, A, R, R_explore, MMC_R, MMC_R_explore, S1, T, M1, M2

    def size(self):
        if self.filled:
            return self.capacity
        else:
            return self.t
