

class ToyMRAbstraction(object):

    def __init__(self, environment, use_sectors=False):
        self.env = environment
        self.use_sectors = use_sectors

    def oo_abstraction_function(self, state):
        if self.use_sectors:
            return self.env.oo_sector_abstraction(state)
        else:
            return self.env.oo_abstraction(state)

    def predicate_func(self, l1_state):
        if self.use_sectors:
            return self.env.sector_predicate_func(l1_state)
        else:
            return self.env.predicate_func(l1_state)
