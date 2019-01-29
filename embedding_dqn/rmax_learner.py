import interfaces
from collections import deque
import numpy as np
import tensorflow as tf
import sys
from abstraction_tools.abstraction_interfaces import L1Action, AbstractState

import l0_learner
from embedding_dqn import value_iteration


class MovingAverageTable(object):

    def __init__(self, moving_avg_len, num_conf, rmax):
        self.moving_avg_len = moving_avg_len
        self.num_conf = num_conf
        self.rmax = rmax

        self.sa_count = dict()
        self.transition_table = dict()
        self.reward_table = dict()
        self.terminal_table = dict()
        self.valid_transitions = dict()

        self.success_table = dict()
        self.success_moving_avg_len = 20

    def insert_action_evaluation(self, action, is_success):
        if action not in self.success_table:
            self.success_table[action] = deque(maxlen=self.success_moving_avg_len)
        self.success_table[action].append(is_success)

    def get_success_rate(self, action):
        if action not in self.success_table or \
                        len(self.success_table[action]) < self.success_moving_avg_len:
            return 0
        return np.mean(self.success_table[action])

    def insert(self, s, a, sp, r, terminal):
        key = (s, a, sp)

        if (s, a) in self.sa_count:
            self.sa_count[(s, a)] += 1
        else:
            self.sa_count[(s, a)] = 0

        if (s, a) in self.valid_transitions:
            self.valid_transitions[(s, a)].add(sp)
        else:
            self.valid_transitions[(s, a)] = {sp}

        if sp not in self.terminal_table:
            self.terminal_table[sp] = deque(maxlen=self.moving_avg_len)
        self.terminal_table[sp].append(float(terminal))

        if key not in self.transition_table:
            self.transition_table[key] = deque(maxlen=self.moving_avg_len)
            self.reward_table[key] = deque(maxlen=self.moving_avg_len)

        for sp_ in self.valid_transitions[(s, a)]:
            self.transition_table[(s, a, sp_)].append(1. if sp == sp_ else 0.)
        self.reward_table[key].append(r)

    def     get_p(self, s, a, sp):
        return np.mean(self.transition_table[(s, a, sp)])

    def get_r(self, s, a, sp, evaluation=False):
        if self.sa_count[(s, a)] >= self.num_conf or evaluation:
            return np.mean(self.reward_table[(s, a, sp)])
        else:
            return self.rmax

    def get_prob_terminal(self, s):
        return np.mean(self.terminal_table[s])

class RMaxLearner(interfaces.LearningAgent):

    def __init__(self, abs_size, env, abs_func, N=1000, max_VI_iterations=100, VI_delta=0.01, value_update_freq=1000, gamma=0.99, rmax=10, max_num_abstract_states=10, frame_history=1):
        self.env = env
        self.abs_size = abs_size
        self.abs_func = abs_func
        self.rmax = rmax
        self.gamma = gamma
        self.utopia_val = self.rmax / (1 - self.gamma)
        self.transition_table = MovingAverageTable(N, 100, self.rmax)
        self.value_iteration = value_iteration.ValueIteration(gamma, max_VI_iterations, VI_delta)
        self.values = dict()
        self.qs = dict()
        self.evaluation_values = dict()
        self.evaluation_qs = dict()
        self.last_evaluation_state = None
        self.last_evaluation_action = None
        self.value_update_counter = 0
        self.value_update_freq = value_update_freq
        self.l0_learner = l0_learner.MultiHeadedDQLearner(abs_size, len(self.env.get_actions_for_state(None)), max_num_abstract_states, frame_history=frame_history, rmax_learner=self)
        self.actions_for_state = dict()
        self.neighbors = dict()
        self.states = set()
        self.current_dqn_number = 0
        self.create_new_state(self.abs_func(self.env.get_current_state()))
        self.run_vi()

    def create_new_state(self, state):
        self.states.add(state)
        self.values[state] = 0
        self.evaluation_values[state] = 0
        self.actions_for_state[state] = [L1Action(state, None, state.get_vector(), state.get_vector(), dqn_number=-1)]
        self.neighbors[state] = []

        print 'Found new state: %s' % (state,)

    def add_new_action(self, state, goal_state):
        new_action = L1Action(state, goal_state, state.get_vector(), goal_state.get_vector(), dqn_number=self.current_dqn_number)
        self.actions_for_state[state].append(new_action)
        self.neighbors[state].append(goal_state)
        self.current_dqn_number += 1

        print 'Found new action: %s' % (new_action,)

    def run_vi(self, evaluation=False):
        transitions = dict()

        for s in self.states:
            for a in self.actions_for_state[s]:
                transitions_sa = []
                key = (s, a)
                if key in self.transition_table.valid_transitions:
                    Z = np.sum(
                        [self.transition_table.get_p(s, a, sp) for sp in self.transition_table.valid_transitions[key]])
                    for sp in self.transition_table.valid_transitions[key]:
                        p = self.transition_table.get_p(s, a, sp) / Z
                        r = self.get_reward(s, a, sp, evaluation=evaluation)
                        t = (1 - self.transition_table.get_prob_terminal(sp))

                        transitions_sa.append((sp, p, r, t))
                transitions[(s, a)] = transitions_sa

        if evaluation:
            self.evaluation_values, self.evaluation_qs =\
                self.value_iteration.run_vi(self.evaluation_values, self.states, self.actions_for_state, transitions, 0)
        else:
            self.values, self.qs =\
                self.value_iteration.run_vi(self.values, self.states, self.actions_for_state, transitions, self.utopia_val)

    def get_reward(self, s, a, sp, evaluation=False):
        # if evaluation:
        #     return self.transition_table.get_r(s, a, sp, evaluation=evaluation)
        # else:
        #     prop = self.l0_learner.replay_buffer.abstract_action_proportions(self.abs_vec_func(s), self.abs_vec_func(sp))
        #     return max(0, 1./len(self.transition_table.actions) - prop)
        return self.transition_table.get_r(s, a, sp, evaluation=evaluation)

    def run_learning_episode(self, environment):
        total_episode_steps = 0
        total_reward = 0

        while not self.env.is_current_state_terminal():

            s = self.abs_func(self.env.get_current_state())

            # need to do additional check here because it is possible to "teleport" without transitioning into a new state
            # to recreate:
            '''
            pick up the key for the first time
            jump off the ledge.
            the game will stop logging "murked" states during your fall, then a terminal state will be called,
            and you will "teleport" into a new state (1, 2) with the key, without having transitioned.
            '''
            if s not in self.states:
                self.create_new_state(s)
                self.run_vi()

            a = self.get_l1_action(s)
            
            dqn_tuple = (a.initial_state, a.goal_state)
            assert s == a.initial_state
            if a.goal_state is not None and np.random.uniform(0, 1) < 0.1:
                eval_action = True
                epsilon = self.l0_learner.epsilon_min
                sys.stdout.write('Executing action: %s -- EVAL ... ' % (a))
            else:
                eval_action = False
                epsilon = max(self.l0_learner.epsilon_min, 1 - self.transition_table.get_success_rate(a))
                sys.stdout.write('Executing action: %s -- eps: %.6f ... ' % (a, epsilon))
            
            episode_steps, R, sp = self.l0_learner.run_learning_episode(self.env, a.initial_state_vec, a.goal_state_vec, s, a.goal_state, a.dqn_number, self.abs_func, epsilon, max_episode_steps=500)
            
            if eval_action:
                self.transition_table.insert_action_evaluation(a, a.goal_state == sp)
            if a.goal_state == sp:
                sys.stdout.write('SUCCESS\n')
            else:
                sys.stdout.write('FAILURE\n')

            # #TODO: REMOVE LATER
            # abs_state = self.env.abstraction_tree.get_abstract_state()
            # in_good_sectors = abs_state.sector in [(1, 2), (1, 1), (2, 1)]
            # if not in_good_sectors:
            #     sp = s

            # Check if finished exploring
            if a.goal_state == None:
                # if self.l0_learner.epsilon[dqn_tuple] <= self.l0_learner.epsilon_min:
                if (s, a) in self.transition_table.sa_count and \
                             self.transition_table.sa_count[(s, a)] >= self.transition_table.num_conf and \
                             len(self.actions_for_state[a.initial_state]) > 1:
                    self.actions_for_state[a.initial_state].remove(a)
                    self.run_vi()

            total_episode_steps += episode_steps
            total_reward += R

            # check transition
            if sp != s:
                if sp not in self.states:
                    self.create_new_state(sp)
                    self.run_vi()

                if sp not in self.neighbors[s]:
                    self.add_new_action(s, sp)
                    self.run_vi()

            # add transition
            self.transition_table.insert(s, a, sp, R, environment.is_current_state_terminal())

            # perform vi for both evaluation values and regular values.
            if self.value_update_counter % self.value_update_freq == 0:
                self.run_vi()
            self.value_update_counter += 1

        return total_episode_steps, total_reward

    def get_l1_action(self, state, evaluation=False):
        if evaluation:
            qs = self.evaluation_qs[state]
        else:
            qs = self.qs[state]
        keys, values = zip(*qs.items())
        # if evaluation:
        #     action = np.random.choice(np.array(keys)[np.array(values) == np.max(values)])
        # else:
        #     temp = 1.0
        #     norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))
        #     distribution = np.exp(temp*norm_values) / np.sum(np.exp(temp*norm_values))
        #     action = keys[np.random.choice(range(len(distribution)), p=distribution)]
        action = np.random.choice(np.array(keys)[np.array(values) == np.max(values)])
        return action

    def get_action(self, state, evaluation=False):

        l1_state = self.abs_func(state)

        if l1_state == self.last_evaluation_state:
            l1_action = self.last_evaluation_action
        else:
            # check if we've ever seen this state
            if l1_state not in self.states:
                return 0

            l1_action = self.get_l1_action(l1_state, evaluation=evaluation)
            self.last_evaluation_state = l1_state
            self.last_evaluation_action = l1_action

        return self.l0_learner.get_action(state, l1_action.initial_state_vec, l1_action.goal_state_vec, l1_action.dqn_number)

    def save_network(self, file_name):
        self.l0_learner.save_network(file_name)
