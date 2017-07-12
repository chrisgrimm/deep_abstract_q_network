import interfaces
from collections import deque
import numpy as np
import sys

from embedding_dqn import oo_l0_learner


class L1ExploreAction(object):
    def __init__(self, attrs, key, pred):
        self.key = key
        self.val = dict(attrs)[key]
        self.pred = pred
        self.count = 0

    def __str__(self):
        return '%s, %s EXPLORE' % (self.key, self.val)

    def get_uid(self):
        return (self.key, self.val) + self.pred + ('explore',)

    def __hash__(self):
        return hash(self.get_uid())

    def __eq__(self, other):
        if not isinstance(other, L1ExploreAction):
            return False
        else:
            return self.get_uid() == other.get_uid()

    def __ne__(self, other):
        return not self.__eq__(other)


def make_diff(attrs1, attrs2):
    assert [a[0] for a in attrs1] == [a[0] for a in attrs2]
    att_dict = {key1 : (val1, val2)
                 for (key1, val1), (key2, val2) in zip(attrs1, attrs2)
                 if val1 != val2}
    return tuple(sorted(att_dict.items()))


def apply_diff(attrs, diff):
    new_attrs = dict(attrs)
    for key, (value1, value2) in diff:
        new_attrs[key] = value2
    return tuple(sorted(new_attrs.items()))

def does_diff_apply(attrs, diff):
    attrs_dict = dict(attrs)
    for key, (value1, value2) in diff:
        if value1 != attrs_dict[key]:
            return False
    return True


class L1Action(object):
    def __init__(self, attrs1, attrs2, pred, dqn_number=0):
        self.diff = make_diff(attrs1, attrs2)
        self.pred = pred
        self.dqn_number = dqn_number

    def __str__(self):
        s = ''
        for key, (att1, att2) in self.diff:
            s += '(%s: %s -> %s) ' % (key, att1, att2)
        return s

    def get_uid(self):
        return self.diff, self.pred

    def __hash__(self):
        return hash(self.get_uid())

    def __eq__(self, other):
        if not isinstance(other, L1Action):
            return False
        else:
            return self.get_uid() == other.get_uid()

    def __ne__(self, other):
        return not self.__eq__(other)


class MovingAverageTable(object):

    def __init__(self, moving_avg_len, num_conf, pred_func):
        self.moving_avg_len = moving_avg_len
        self.num_conf = num_conf

        self.pred_func = pred_func

        self.a_count = dict()
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

        diff = make_diff(s, sp)

        key = (diff, a)

        if a in self.a_count:
            self.a_count[a] += 1
        else:
            self.a_count[a] = 1

        if a in self.valid_transitions:
            self.valid_transitions[a].add(diff)
        else:
            self.valid_transitions[a] = {diff}

        if key not in self.terminal_table:
            self.terminal_table[key] = deque(maxlen=self.moving_avg_len)
        self.terminal_table[key].append(float(terminal))

        if key not in self.transition_table:
            self.transition_table[key] = deque(maxlen=self.moving_avg_len)
            self.reward_table[key] = deque(maxlen=self.moving_avg_len)

        for diff_ in self.valid_transitions[a]:
            self.transition_table[(diff_, a)].append(1. if diff == diff_ else 0.)
        self.reward_table[key].append(r)

    def get_p(self, s, a, sp):
        diff = make_diff(s, sp)

        return np.mean(self.transition_table[(diff, a)])

    def get_r(self, s, a, sp, evaluation=False):
        diff = make_diff(s, sp)

        return np.mean(self.reward_table[(diff, a)])

    def get_prob_terminal(self, s, a, sp):
        diff = make_diff(s, sp)

        return np.mean(self.terminal_table[(diff, a)])


class OORMaxLearner(interfaces.LearningAgent):
    def __init__(self, abs_size, env, abs_func, pred_func, N=1000, max_VI_iterations=100, VI_delta=0.01, gamma=0.99, rmax=1,
                 max_num_abstract_states=10, frame_history=1):
        self.env = env
        self.abs_size = abs_size
        self.abs_func = abs_func
        self.pred_func = pred_func  # takes in abstract state -> outputs predicates dict
        self.rmax = rmax
        self.gamma = gamma
        self.utopia_val = self.rmax / (1 - self.gamma)
        self.transition_table = MovingAverageTable(N, 100, pred_func)
        self.max_VI_iterations = max_VI_iterations
        self.VI_delta = VI_delta
        self.values = dict()
        self.evaluation_values = dict()
        self.value_update_counter = 0
        self.value_update_freq = 10
        self.l0_learner = oo_l0_learner.MultiHeadedDQLearner(abs_size, len(self.env.get_actions_for_state(None)),
                                                          max_num_abstract_states, frame_history=frame_history,
                                                          rmax_learner=self)
        self.actions_for_pia = dict()  # pia = (predicates, key, attribute)
        self.explore_for_pia = dict()  # holds the reference to all the explore actions
        self.neighbors = dict()  # indexed by pia
        self.actions = set()  # used to check if actions already exist
        self.states = set()
        self.current_dqn_number = 0
        self.create_new_state(self.abs_func(self.env.get_current_state()))

    def create_new_state(self, state):
        self.states.add(state)
        self.values[state] = 0
        self.evaluation_values[state] = 0

        # create explore actions for each attribute:
        preds = self.pred_func(state)
        for i, val in state:
            pia = (preds, i, val)
            if pia not in self.actions_for_pia:
                explore_action = L1ExploreAction(state, i, preds)
                self.actions_for_pia[pia] = [explore_action]
                self.explore_for_pia[pia] = explore_action
                self.neighbors[pia] = []

        print 'Found new state: %s' % (state,)

    def add_new_action(self, state, goal_state):

        preds = self.pred_func(state)
        goal_diff = make_diff(state, goal_state)
        new_action = L1Action(state, goal_state, preds, dqn_number=self.current_dqn_number)

        # Check if action already exists
        if new_action in self.actions:
            return

        for i, (att, att_goal) in goal_diff:
            pia = (preds, i, att)
            goal_pia = (preds, i, att_goal)
            self.actions.add(new_action)
            self.actions_for_pia[pia].append(new_action)
            self.neighbors[pia].append(goal_pia)

        self.current_dqn_number += 1

        print 'Found new action: %s' % (new_action,)

    def populate_imagined_states(self):
        old_states = dict()
        while len(old_states) != len(self.states):
            old_states = self.states.copy()
            for s in old_states:
                for a in self.get_all_actions_for_state(s):
                    if a in self.transition_table.valid_transitions:
                        for diff in self.transition_table.valid_transitions[a]:
                            if not does_diff_apply(s, diff):
                                continue
                            sp = apply_diff(s, diff)
                            if sp not in self.states:
                                self.create_new_state(sp)

    def run_vi(self, values, evaluation=False):
        new_values = dict()
        for i in xrange(self.max_VI_iterations):
            stop = True
            for s in self.states:
                new_values[s] = np.max(self.calculate_qs(s, values, evaluation=evaluation).values())

                if s in values and np.abs(new_values[s] - values[s]) > self.VI_delta:
                    stop = False
            values = new_values.copy()
            if stop:
                break
        return values

    def get_all_actions_for_state(self, state):
        actions = []
        preds = self.pred_func(state)
        for i, att in state:
            pia = (preds, i, att)
            actions.extend(self.actions_for_pia[pia])
        return actions

    def calculate_qs(self, s, values, evaluation=False):
        qs = dict()

        for a in self.get_all_actions_for_state(s):
            # when evaluating dont use rmax for underexplored states, for invalid transitions assign 0-value.
            if a in self.transition_table.a_count:
                if self.transition_table.a_count[a] >= self.transition_table.num_conf or evaluation:
                    val = 0

                    if a in self.transition_table.valid_transitions:
                        Z = np.sum(
                            [self.transition_table.get_p(s, a, apply_diff(s, diff)) for diff in self.transition_table.valid_transitions[a] if does_diff_apply(s, diff)])
                        for diff in self.transition_table.valid_transitions[a]:
                            if not does_diff_apply(s, diff):
                                continue

                            sp = apply_diff(s, diff)
                            p = self.transition_table.get_p(s, a, sp) / Z
                            r = self.get_reward(s, a, sp, evaluation=evaluation)

                            if sp in values:
                                use_backup = (1 - self.transition_table.get_prob_terminal(s, a, sp))
                                val += p * (r + self.gamma * values[sp] * use_backup)
                            else:
                                val += p * (r + self.gamma * self.utopia_val)
                else:
                    val = self.utopia_val
            else:
                if evaluation:
                    val = 0
                else:
                    val = self.utopia_val
            qs[a] = val
        return qs

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
                self.populate_imagined_states()
                self.values = self.run_vi(self.values.copy())

            a = self.get_l1_action(s)

            if type(a) is not L1ExploreAction and np.random.uniform(0, 1) < 0.1:
                eval_action = True
                epsilon = self.l0_learner.epsilon_min
                sys.stdout.write('Executing action: %s -- EVAL ... ' % (a))
            else:
                eval_action = False
                epsilon = max(self.l0_learner.epsilon_min, 1 - self.transition_table.get_success_rate(a))
                sys.stdout.write('Executing action: %s -- eps: %.6f ... ' % (a, epsilon))

            if type(a) is L1ExploreAction:
                s_goal = None
                dqn_number = -1
            else:
                s_goal = apply_diff(s, a.diff)
                dqn_number = a.dqn_number

            episode_steps, R, sp = self.l0_learner.run_learning_episode(self.env, s, s_goal,
                                                                        dqn_number, self.abs_func, epsilon,
                                                                        max_episode_steps=500)

            if type(a) is not L1ExploreAction:
                true_diff = make_diff(s, sp)
                is_success = a.diff == true_diff
            else:
                is_success = False

            if eval_action:
                self.transition_table.insert_action_evaluation(a, is_success)
            if is_success:
                sys.stdout.write('SUCCESS\n')
            else:
                sys.stdout.write('FAILURE\n')

            # #TODO: REMOVE LATER
            # abs_state = self.env.abstraction_tree.get_abstract_state()
            # in_good_sectors = abs_state.sector in [(1, 2), (1, 1), (2, 1)]
            # if not in_good_sectors:
            #     sp = s

            # Check if finished exploring
            if type(a) is L1ExploreAction:
                preds = self.pred_func(s)
                for i, val in s:
                    pia = (preds, i, val)
                    explore_action = self.explore_for_pia[pia]
                    explore_action.count += 1
                    # remove the explore action for this attribute if its been explored enough
                    if explore_action in self.actions_for_pia[pia] and \
                            explore_action.count >= self.transition_table.num_conf and \
                            len(self.get_all_actions_for_state(s)) > 1:
                        self.actions_for_pia[pia].remove(explore_action)

            total_episode_steps += episode_steps
            total_reward += R

            # check transition
            if sp != s:
                self.add_new_action(s, sp)

            # add transition
            self.transition_table.insert(s, a, sp, R, environment.is_current_state_terminal())

            # perform vi for both evaluation values and regular values.
            if self.value_update_counter % self.value_update_freq == 0:
                self.values = self.run_vi(self.values.copy())
            self.value_update_counter += 1

        return total_episode_steps, total_reward

    def get_l1_action(self, state, evaluation=False):
        values = self.evaluation_values if evaluation else self.values
        qs = self.calculate_qs(state, values, evaluation=evaluation)
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

        # check if we've ever seen this state
        if l1_state not in self.states:
            return 0

        l1_action = self.get_l1_action(l1_state, evaluation=evaluation)

        if type(l1_action) is L1ExploreAction:
            return np.random.choice(self.env.get_actions_for_state(state))
        else:
            return self.l0_learner.get_action(state, l1_action.dqn_number)

    def save_network(self, file_name):
        self.l0_learner.save_network(file_name)
