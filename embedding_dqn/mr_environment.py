import atari

class MREnvironment(atari.AtariEnvironment):

    def reset_environment(self):
        # no-ops once the agent is about to get murked.
        # loop will hang if we dont check that lives > 0
        while self.abstraction_tree.bout_to_get_murked() and self.current_lives > 0:
            self.perform_action(0)

        super(MREnvironment, self).reset_environment()

    # def perform_action(self, onehot_index_action):
    #     state, atari_action, reward, next_state, is_terminal = super(MREnvironment, self).perform_action(onehot_index_action)
    #
    #     # abs_state = self.abstraction_tree.get_abstract_state()
    #     # in_good_sectors = abs_state.sector in [(1,2), (1,1), (2,1)]
    #     # self.is_terminal = self.is_terminal or not in_good_sectors
    #
    #     return state, atari_action, reward, next_state, self.is_terminal