import numpy as np
from embedding_dqn.rmax_learner import AbstractState

room_index = 3

global_object_index = 67
global_object_locs = [  # stored as (byte, bit)
    (0, 0),  # Key Room 1
    (3, 0),  # Key Room 7
    (4, 4),  # Key Room 8
    (7, 4),  # Key Room 14
    (0, 2),  # Door R Room 1
    (0, 3),  # Door L Room 1
    (2, 2),  # Door R Room 5
    (0, 3),  # Door L Room 5
    (8, 2),  # Door R Room 17
    (8, 3),  # Door L Room 17
]


def get_bit(a, i):
    return a & (2**i) != 0


class MRAbstraction(object):

    def __init__(self):
        self.global_state = [0] * len(global_object_locs)
        self.current_room = 1
        self.agent_sector = (1, 2)
        self.env = None

    def set_env(self, env):
        self.env = env

    def update_global_state(self, ram):
        for i, (byte, bit) in enumerate(global_object_locs):
            self.global_state[i] = get_bit(ram[global_object_index + byte], bit)

    def should_perform_sector_check(self, ram):
        is_falling = ram[88] > 0
        death_counter_active = ram[55] > 0
        death_sprite_active = ram[54] == 6
        is_walking_or_on_stairs = ram[53] in [0, 10, 8, 18]
        # also need landing because there is a frame where the agent is walking when it jumps on the rope
        landing = ram[91] > 0
        should_check = (is_walking_or_on_stairs and not landing and not is_falling and not death_counter_active and not death_sprite_active)
        return should_check

    def bout_to_get_murked(self, ram):
        is_falling = ram[88] > 0
        death_counter_active = ram[55] > 0
        death_sprite_active = ram[54] == 6
        return is_falling or death_counter_active or death_sprite_active

    def update_agent_sector(self, ram):
        if self.should_perform_sector_check(ram):
            pos_x, pos_y = (ram[0xAA - 0x80] - 0x01) / float(0x98 - 0x01), (ram[0xAB - 0x80] - 0x86) / float(0xFF - 0x86)
            self.agent_sector = np.clip(int(3 * pos_x), 0, 2), np.clip(int(3 * pos_y), 0, 2)
        return self.agent_sector

    def update_current_room(self, ram):
        self.current_room = ram[room_index]

    def update_state(self, ram):
        self.update_global_state(ram)
        self.update_agent_sector(ram)
        self.update_current_room(ram)

    def get_abstract_state(self):
        return MRAbstractState(self.current_room, self.agent_sector, self.global_state)

    def abstraction_function(self, x):
        self.update_state(self.env.getRAM())
        return self.get_abstract_state()

class MRAbstractState(AbstractState):

    def __init__(self, room, agent_sector, global_state):
        self.room = room
        self.agent_sector = agent_sector
        self.global_state = tuple(global_state)

    def get_key_lazy(self):
        return (self.room,) + self.agent_sector + self.global_state

    def get_vector_lazy(self):
        onehot_room = [0] * 24
        onehot_room[self.room] = 1
        onehot_sector = [0] * 9
        onehot_sector[3*self.agent_sector[1] + self.agent_sector[0]] = 1
        posneg_global_state = [1 if x == True else -1 for x in self.global_state]
        return onehot_room + onehot_sector + posneg_global_state

    def __str__(self):
        bools = ''.join(str(int(b)) for b in self.global_state)
        return '%s - %s - %s' % (str(self.room), str(self.agent_sector), bools)