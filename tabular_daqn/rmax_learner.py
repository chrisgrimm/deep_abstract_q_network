import interfaces
from collections import deque
import numpy as np

from l0_learner import MultiHeadedDQLearner


class L1Action(object):
    def __init__(self, initial_state, goal_state, dqn_number):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.dqn_number = dqn_number

    def __str__(self):
        if self.goal_state is None:
            return '%s EXPLORE' % (self.initial_state,)
        else:
            return '%s -> %s' % (self.initial_state, self.goal_state)

class MovingAverageTable(object):

    def __init__(self, moving_avg_len, num_conf, rmax):
        self.moving_avg_len = moving_avg_len
        self.num_conf = num_conf
        self.rmax = rmax

        self.sa_count = dict()
        self.transition_table = dict()
        self.reward_table = dict()
        self.valid_transitions = dict()
        self.states = set()
        self.terminal_states = set()
        self.actions = set()

    def insert(self, s, a, sp, r, terminal):
        self.states.add(s)
        self.actions.add(a)
        key = (s, a, sp)

        if (s, a) in self.sa_count:
            self.sa_count[(s, a)] += 1
        else:
            self.sa_count[(s, a)] = 0

        if (s, a) in self.valid_transitions:
            self.valid_transitions[(s, a)].add(sp)
        else:
            self.valid_transitions[(s, a)] = {sp}

        if key not in self.transition_table:
            self.transition_table[key] = deque(maxlen=self.moving_avg_len)
            self.reward_table[key] = deque(maxlen=self.moving_avg_len)

        for sp_ in self.valid_transitions[(s, a)]:
            self.transition_table[(s, a, sp_)].append(1. if sp == sp_ else 0.)
        self.reward_table[key].append(r)

    def get_p(self, s, a, sp):
        return np.mean(self.transition_table[(s, a, sp)])

    def get_r(self, s, a, sp):
        if self.sa_count[(s, a)] >= self.num_conf:
            return np.mean(self.reward_table[(s, a, sp)])
        else:
            return self.rmax

    def is_terminal(self, s):
        return s in self.terminal_states


class RMaxLearner(interfaces.LearningAgent):

    def __init__(self, env, abs_func, N=100, max_VI_iterations=100, VI_delta=0.01, gamma=0.9, rmax=10, max_num_abstract_states=10, frame_history=1):
        self.env = env
        self.abs_func = abs_func
        self.rmax = rmax
        self.transition_table = MovingAverageTable(N, 100, self.rmax)
        self.max_VI_iterations = max_VI_iterations
        self.VI_delta = VI_delta
        self.values = dict()
        self.gamma = gamma

        self.l0_learner = MultiHeadedDQLearner(len(self.env.get_actions_for_state(None)), max_num_abstract_states, frame_history=frame_history)
        self.actions_for_state = dict()
        self.neighbors = dict()
        self.states = set()
        self.current_dqn_number = 0
        self.create_new_state(self.abs_func(self.env.get_current_state()))

    def create_new_state(self, state):
        self.states.add(state)
        self.values[state] = 0
        self.actions_for_state[state] = [L1Action(state, None, self.current_dqn_number)]
        self.neighbors[state] = []
        self.current_dqn_number += 1

        print 'Found new state: %s' % (state,)

    def add_new_action(self, state, goal_state):
        new_action = L1Action(state, goal_state, self.current_dqn_number)
        self.actions_for_state[state].append(new_action)
        self.neighbors[state].append(goal_state)
        self.current_dqn_number += 1

        print 'Found new action: %s' % (new_action,)

    def run_vi(self):
        new_values = dict()
        for i in xrange(self.max_VI_iterations):
            stop = True
            for s in self.transition_table.states:
                new_values[s] = np.max(self.calculate_qs(s).values())
                if np.abs(new_values[s] - self.values[s]) > self.VI_delta:
                    stop = False
            if stop:
                break

    def calculate_qs(self, s):
        qs = dict()
        for a in self.actions_for_state[s]:
            val = 0

            key = (s, a)
            if key in self.transition_table.valid_transitions:
                for sp in self.transition_table.valid_transitions[key]:
                    p = self.transition_table.get_p(s, a, sp)
                    r = self.transition_table.get_r(s, a, sp)
                    use_backup = (1 - self.transition_table.is_terminal(sp))
                    val += p * (r + self.gamma * self.values[sp] * use_backup)
            else:
                val = self.rmax
            qs[a] = val
        return qs

    def run_learning_episode(self, environment):
        total_episode_steps = 0
        total_reward = 0

        while not self.env.is_current_state_terminal():

            s = self.abs_func(self.env.get_current_state())

            a = self.get_action(s)

            assert s == a.initial_state
            print 'Executing action: %s -- eps: %.6f' % (a, self.l0_learner.epsilon[a.dqn_number])
            episode_steps, R, sp = self.l0_learner.run_learning_episode(self.env, a.dqn_number, s, a.goal_state, self.abs_func, max_episode_steps=1000)

            total_episode_steps += episode_steps
            total_reward += R

            # check transition
            if sp != s:
                if sp not in self.states:
                    self.create_new_state(sp)

                if sp not in self.neighbors[s]:
                    self.add_new_action(s, sp)

            # add transition
            self.transition_table.insert(s, a, sp, R, environment.is_current_state_terminal())

            # perform vi
            self.run_vi()

        return total_episode_steps, total_reward

    def get_action(self, state):
        qs = self.calculate_qs(state)
        keys, values = zip(*qs.items())
        action = np.random.choice(np.array(keys)[np.array(values) == np.max(values)])
        return action