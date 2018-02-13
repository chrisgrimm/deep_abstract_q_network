from environments import atari

room_index = 3

class MREnvironment(atari.AtariEnvironment):

    def __init__(self, atari_rom, frame_skip=4, noop_max=30, terminate_on_end_life=False, random_seed=123,
                 frame_history_length=4, use_gui=False, max_num_frames=500000, repeat_action_probability=0.0, single_life=False,
                 record_screen_dir=None):
        super(MREnvironment, self).__init__(atari_rom, frame_skip, noop_max, terminate_on_end_life, random_seed,
                 frame_history_length, use_gui, max_num_frames, repeat_action_probability, record_screen_dir)
        self.discovered_rooms = set()
        self.single_life = single_life
        self.abstraction = None

    def perform_action(self, onehot_index_action):
        state, atari_action, reward, next_state, is_terminal = super(MREnvironment, self).perform_action(onehot_index_action)

        new_room = self.getRAM()[room_index]
        self.discovered_rooms.add(new_room)

        return state, atari_action, reward, next_state, self.is_terminal

    def get_discovered_rooms(self):
        return self.discovered_rooms

    def _act(self, ale_action, repeat):
        if self.single_life:
            self.terminate_on_end_life = True
        return super(MREnvironment, self)._act(ale_action, repeat)

    def reset_environment(self):
        if self.single_life:
            self.terminate_on_end_life = False
        super(MREnvironment, self).reset_environment()

        # if self.abstraction is None:
        #     if self.single_life:
        #         self.terminate_on_end_life = False
        #     super(MREnvironment, self).reset_environment()
        # else:
        #     # no-ops once the agent is about to get murked.
        #     # loop will hang if we dont check that lives > 0
        #     while not self.abstraction.should_perform_sector_check(self.getRAM()) and not self.ale.game_over():
        #         self.perform_action(0)
        #
        #     if self.single_life:
        #         self.terminate_on_end_life = False
        #     super(MREnvironment, self).reset_environment()
        #     if self.current_lives >= 6:
        #         self.abstraction.reset(self.getRAM())

    # def set_abstraction(self, abstraction):
    #     self.abstraction = abstraction
    #
    # def reset_environment(self):
    #     # no-ops once the agent is about to get murked.
    #     # loop will hang if we dont check that lives > 0
    #     while not self.abstraction.should_perform_sector_check(self.getRAM()) and not self.ale.game_over():
    #         self.perform_action(0)
    #
    #     super(MREnvironment, self).reset_environment()
    #     if self.current_lives >= 6:
    #         self.abstraction.reset(self.getRAM())

    # def perform_action(self, onehot_index_action):
    #     state, atari_action, reward, next_state, is_terminal = super(MREnvironment, self).perform_action(onehot_index_action)
    #
    #     # abs_state = self.abstraction_tree.get_abstract_state()
    #     # in_good_sectors = abs_state.sector in [(1,2), (1,1), (2,1)]
    #     # self.is_terminal = self.is_terminal or not in_good_sectors
    #
    #     return state, atari_action, reward, next_state, self.is_terminal