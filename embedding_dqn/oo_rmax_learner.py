import interfaces
from collections import deque
import numpy as np
import sys
import dill

from cts import cpp_cts
from embedding_dqn import oo_l0_learner
from embedding_dqn import value_iteration


class L1ExploreAction(object):
    def __init__(self, attrs, key, pred, dqn_number=-1):
        self.key = key
        self.val = dict(attrs)[key]
        self.pred = pred
        self.count = 0
        self.dqn_number = dqn_number

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
    def __init__(self, attrs1, attrs2, pred, dqn_number=-1):
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

    def __init__(self, num_conf, pred_func, moving_avg_len=None):
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

        if self.moving_avg_len is None:
            if key not in self.terminal_table:
                self.terminal_table[key] = 0
            self.terminal_table[key] += 1

            if key not in self.transition_table:
                self.transition_table[key] = 0
                self.reward_table[key] = 0

            self.transition_table[key] += 1
            self.reward_table[key] += r
        else:
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

    def get_r(self, s, a, sp):
        diff = make_diff(s, sp)

        return np.mean(self.reward_table[(diff, a)])

    def get_prob_terminal(self, s, a, sp):
        diff = make_diff(s, sp)

        if self.moving_avg_len is None:
            return self.terminal_table[(diff, a)]/self.transition_table[(diff, a)]
        else:
            return np.mean(self.terminal_table[(diff, a)])


class OORMaxLearner(interfaces.LearningAgent):
    def __init__(self, abs_size, env, abs_func, pred_func, N=1000, max_VI_iterations=100, value_update_freq=1000, VI_delta=0.01, gamma=0.99, rmax=1,
                 max_num_abstract_states=10, frame_history=1, restore_file=None, error_clip=1,
                 state_encoder=None, bonus_beta=0.05, cts_size=None):
        self.env = env
        self.abs_size = abs_size
        self.abs_func = abs_func
        self.pred_func = pred_func  # takes in abstract state -> outputs predicates dict
        self.rmax = rmax
        self.gamma = gamma
        self.utopia_val = self.rmax / (1 - self.gamma)
        self.transition_table = MovingAverageTable(100, pred_func)
        self.value_iteration = value_iteration.ValueIteration(gamma, max_VI_iterations, VI_delta)
        self.values = dict()
        self.qs = dict()
        self.evaluation_values = dict()
        self.evaluation_qs = dict()
        self.last_evaluation_state = None
        self.last_evaluation_action = None
        self.value_update_counter = 0
        self.value_update_freq = value_update_freq

        restore_network_file = None
        if restore_file is not None:
            restore_network_file = 'oo_net.ckpt'
        self.l0_learner = oo_l0_learner.MultiHeadedDQLearner(abs_size, len(self.env.get_actions_for_state(None)),
                                                            max_num_abstract_states, frame_history=frame_history,
                                                            rmax_learner=self, restore_network_file=restore_network_file,
                                                            encoding_func=state_encoder, bonus_beta=bonus_beta,
                                                            error_clip=error_clip)
        self.actions_for_pia = dict()  # pia = (predicates, key, attribute)
        self.explore_for_pia = dict()  # holds the reference to all the explore actions
        self.actions = set()  # used to check if actions already exist
        self.states = set()
        self.current_dqn_number = 1

        self.cts = dict()
        self.global_cts = cpp_cts.CPP_CTS(*cts_size)
        self.encoding_func = state_encoder
        self.bonus_beta = bonus_beta
        self.cts_size = cts_size
        self.using_global_epsilon = False # state_encoder is not None

        if restore_file is None:
            self.create_new_state(self.abs_func(self.env.get_current_state()))
        else:
            with open('./transition_table.pickle', 'r') as f:
                self.transition_table = dill.load(f)
            with open('./states.pickle', 'r') as f:
                self.states = dill.load(f)
            with open('./actions.pickle', 'r') as f:
                self.actions = dill.load(f)
            with open('./actions_for_pia.pickle', 'r') as f:
                self.actions_for_pia = dill.load(f)
            with open('./explore_for_pia.pickle', 'r') as f:
                self.explore_for_pia = dill.load(f)
            self.populate_imagined_states()
        self.run_vi()

    def create_new_state(self, state):
        self.states.add(state)
        self.values[state] = self.utopia_val
        self.evaluation_values[state] = 0

        # create explore actions for each attribute:
        preds = self.pred_func(state)
        for i, val in state:
            pia = (preds, i, val)
            if pia not in self.actions_for_pia:
                explore_action = L1ExploreAction(state, i, preds, dqn_number=0)
                # self.current_dqn_number += 1

                if self.encoding_func is not None:
                    self.cts[explore_action] = cpp_cts.CPP_CTS(*self.cts_size)

                self.actions_for_pia[pia] = [explore_action]
                self.explore_for_pia[pia] = explore_action
                self.actions.add(explore_action)

        print 'Found new state: %s' % (state,)

    def add_new_action(self, state, goal_state):

        preds = self.pred_func(state)
        goal_diff = make_diff(state, goal_state)
        new_action = L1Action(state, goal_state, preds, dqn_number=self.current_dqn_number)

        # Check if action already exists
        if new_action in self.actions:
            return

        # The action does not exist, so add it

        if self.encoding_func is not None:
            self.cts[new_action] = cpp_cts.CPP_CTS(*self.cts_size)

        for i, (att, att_goal) in goal_diff:
            pia = (preds, i, att)
            goal_pia = (preds, i, att_goal)
            self.actions.add(new_action)
            self.actions_for_pia[pia].append(new_action)

        self.current_dqn_number += 1

        self.populate_imagined_states()
        self.run_vi()

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

    def run_vi(self, evaluation=False):
        actions_for_state = dict()
        transitions = dict()

        for s in self.states:
            actions = self.get_all_actions_for_state(s)
            actions_for_state[s] = actions
            for a in actions:
                transitions_sa = []
                if a in self.transition_table.a_count and \
                        (self.transition_table.a_count[a] >= self.transition_table.num_conf or evaluation) and \
                                a in self.transition_table.valid_transitions:
                    Z = np.sum(
                        [self.transition_table.get_p(s, a, apply_diff(s, diff)) for diff in
                         self.transition_table.valid_transitions[a] if does_diff_apply(s, diff)])
                    for diff in self.transition_table.valid_transitions[a]:
                        if not does_diff_apply(s, diff):
                            continue

                        sp = apply_diff(s, diff)
                        p = self.transition_table.get_p(s, a, sp) / Z
                        r = self.get_reward(s, a, sp)
                        t = self.transition_table.get_prob_terminal(s, a, sp)

                        transitions_sa.append((sp, p, r, t))
                transitions[(s, a)] = transitions_sa

        if evaluation:
            self.evaluation_values, self.evaluation_qs =\
                self.value_iteration.run_vi(self.evaluation_values, self.states, actions_for_state, transitions, 0)
        else:
            self.values, self.qs =\
                self.value_iteration.run_vi(self.values, self.states, actions_for_state, transitions, self.utopia_val)

    def get_all_actions_for_state(self, state):
        actions = []
        preds = self.pred_func(state)
        for i, att in state:
            pia = (preds, i, att)
            for a in self.actions_for_pia[pia]:
                if type(a) is L1ExploreAction or \
                        does_diff_apply(state, a.diff):
                    actions.append(a)
        return actions

    def get_reward(self, s, a, sp):
        # if evaluation:
        #     return self.transition_table.get_r(s, a, sp, evaluation=evaluation)
        # else:
        #     prop = self.l0_learner.replay_buffer.abstract_action_proportions(self.abs_vec_func(s), self.abs_vec_func(sp))
        #     return max(0, 1./len(self.transition_table.actions) - prop)
        return self.transition_table.get_r(s, a, sp)

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
                self.run_vi()

            a = self.get_l1_action(s)

            if self.using_global_epsilon:
                if a.dqn_number not in self.l0_learner.epsilon:
                    self.l0_learner.epsilon[a.dqn_number] = 1.0
                epsilon = self.l0_learner.epsilon[a.dqn_number]
                # epsilon = self.l0_learner.global_epsilon
                eval_action = True
                sys.stdout.write('Executing action: %s -- eps: %.6f ... ' % (a, epsilon))
            else:
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
            else:
                s_goal = apply_diff(s, a.diff)

            dqn_number = a.dqn_number

            cts = None if self.encoding_func is None else self.cts[a]
            dqn_distribution = None  # self.get_dqn_distribution()

            episode_steps, R, sp = self.l0_learner.run_learning_episode(self.env, s, s_goal, a,
                                                                        dqn_number, self.abs_func, epsilon,
                                                                        max_episode_steps=1000,
                                                                        cts=cts, dqn_distribution=dqn_distribution)

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
                changed = False
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
                        self.actions.remove(explore_action)
                        changed = True
                if changed:
                    self.run_vi()

            total_episode_steps += episode_steps
            total_reward += R

            # check transition
            if sp != s:
                self.add_new_action(s, sp)

            # add transition
            self.transition_table.insert(s, a, sp, R, environment.is_current_state_terminal())

            # perform vi for both evaluation values and regular values.
            if self.value_update_counter % self.value_update_freq == 0:
                self.run_vi()
            self.value_update_counter += 1

        return total_episode_steps, total_reward

    def get_dqn_distribution(self):
        dist = dict()
        epsilon = 0.1
        for a in self.actions:
            if a in self.transition_table.a_count and self.transition_table.a_count[a] >= 1:
                p = 0
                if type(a) is not L1ExploreAction:
                    p = self.transition_table.get_success_rate(a)
                dist[a.dqn_number] = 1 - p + epsilon
        sum = np.sum(dist.values())
        for key in dist:
            dist[key] /= sum
        return dist

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
        action = np.random.choice(np.array(keys)[(np.max(values) - np.array(values)) < 0.00001])
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

        if type(l1_action) is L1ExploreAction:
            return np.random.choice(self.env.get_actions_for_state(state))
        else:
            return self.l0_learner.get_action(state, l1_action.dqn_number)

    def save_network(self, file_name):
        pass
        # self.l0_learner.save_network(file_name)
