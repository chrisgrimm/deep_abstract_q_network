
class Environment(object):

    def perform_action(self, action):
        raise NotImplemented

    def get_current_state(self):
        raise NotImplemented

    def get_actions_for_state(self, state):
        raise NotImplemented

    def reset_environment(self):
        raise NotImplemented

    def is_current_state_terminal(self):
        raise NotImplemented


class LearningAgent(object):

    def run_learning_episode(self, environment):
        raise NotImplemented

    def get_action(self, state):
        raise NotImplemented


class DQNInterface(object):

    def get_input_shape(self):
        raise NotImplemented

    def get_input_dtype(self):
        raise NotImplemented

    def construct_q_network(self, input):
        raise NotImplemented