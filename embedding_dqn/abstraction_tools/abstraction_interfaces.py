class L1Action(object):
    def __init__(self, initial_state, goal_state, initial_state_vec, goal_state_vec, dqn_number=0):
        assert initial_state is None or issubclass(initial_state.__class__, AbstractState)
        assert goal_state is None or issubclass(goal_state.__class__, AbstractState)
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.initial_state_vec = initial_state_vec
        self.goal_state_vec = goal_state_vec
        self.dqn_number = dqn_number

    def __str__(self):
        if self.goal_state is None:
            return '%s EXPLORE' % (self.initial_state,)
        else:
            return '%s -> %s' % (self.initial_state, self.goal_state)

    def get_key(self):
        initial_state_key = () if self.initial_state is None else self.initial_state.get_key()
        goal_state_key = () if self.goal_state is None else self.goal_state.get_key()
        return (initial_state_key, goal_state_key, self.dqn_number)

    def __hash__(self):
        return hash(self.get_key())

    def __eq__(self, other):
        if not isinstance(other, L1Action):
            return False
        else:
            return self.get_key() == other.get_key()


class AbstractState(object):

    def get_key_lazy(self):
        raise NotImplemented

    def get_vector_lazy(self):
        raise NotImplemented

    def get_key(self):
        if hasattr(self, '__key'):
            return self.__key
        else:
            self.__key = self.get_key_lazy()
            return self.__key

    def get_vector(self):
        if hasattr(self, '__vector'):
            return self.__vector
        else:
            self.__vector = self.get_vector_lazy()
            return self.__vector

    def __hash__(self):
        return hash(self.get_key())

    def __eq__(self, other):
        if not issubclass(other.__class__, AbstractState):
            return False
        else:
            return self.get_key() == other.get_key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.get_key())
