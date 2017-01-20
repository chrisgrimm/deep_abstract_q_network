
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


