from environments import atari

room_index = 3

class MREnvironment(atari.AtariEnvironment):

    def __init__(self, config):
        super(MREnvironment, self).__init__(config)

        # Set configuration params
        self.single_life = eval(self.config['ENV']['USE_SINGLE_LIFE'])

        self.discovered_rooms = set()
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
