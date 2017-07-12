import numpy as np
from abstraction_interfaces import AbstractState

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
    (2, 3),  # Door L Room 5
    (8, 2),  # Door R Room 17
    (8, 3),  # Door L Room 17
]


def get_bit(a, i):
    return a & (2**i) != 0


class MRAbstraction(object):

    def __init__(self, use_sectors=True):
        self.global_state = [0] * len(global_object_locs)
        self.env = None
        self.current_room = 1
        self.use_sectors = use_sectors
        self.agent_sector = (1, 2)
        self.old_should_check = True
        self.update_room_value = None
        self.updated_room = False
        self.old_RAM = None

    def reset(self, ram):
        self.old_should_check = True
        self.update_state(ram, hard=True)

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
        new_should_check = (is_walking_or_on_stairs and not is_falling and not death_counter_active and not death_sprite_active)
        should_check = new_should_check and self.old_should_check
        self.old_should_check = new_should_check
        return should_check or self.updated_room
    #
    # def bout_to_get_murked(self, ram):
    #     is_falling = ram[88] > 0
    #     death_counter_active = ram[55] > 0
    #     death_sprite_active = ram[54] == 6
    #     return is_falling or death_counter_active or death_sprite_active

    def get_agent_position(self, ram):
        pos_x, pos_y = (ram[0xAA - 0x80] - 0x01) / float(0x98 - 0x01), (ram[0xAB - 0x80] - 0x86) / float(0xFF - 0x86)
        return pos_x, pos_y

    def update_agent_sector_normal_room(self, ram):
        pos_x, pos_y = self.get_agent_position(ram)
        self.agent_sector = np.clip(int(3 * pos_x), 0, 2), np.clip(int(3 * pos_y), 0, 2)

    def update_agent_sector_water_room(self, ram):
        pos_x, pos_y = self.get_agent_position(ram)
        sectors_room_0 = [(0, 0.2517), (0.2517, 0.7546), (0.7546, 0.912), (0.912, 1.0)]
        sectors_room_7 = [(0, 0.09245), (0.09245, 0.25496), (0.25496, 0.7516), (0.7516, 1.0)]
        sectors_room_12 = [(0, 0.24834), (0.24834, 0.40725), (0.40725, 0.5993), (0.5993, 0.75826), (0.75826, 1.0)]
        room_dict = {0: sectors_room_0, 7: sectors_room_7, 12: sectors_room_12}
        active_room = room_dict[self.current_room]
        for sector, (a, b) in enumerate(active_room):
            if a <= pos_x <= b:
                # HAX
                self.agent_sector = sector % 3, sector/3

    def update_agent_sector(self, ram, hard=False):
        if hard or self.should_perform_sector_check(ram):
            if self.current_room in [0, 7, 12]:
                self.update_agent_sector_water_room(ram)
            else:
                self.update_agent_sector_normal_room(ram)

    def update_current_room(self, ram, hard=False):
        if hard:
            self.current_room = ram[room_index]
            return

        new_room = ram[room_index]
        if self.update_room_value is not None:
            self.current_room = self.update_room_value
            self.updated_room = True
        else:
            self.updated_room = False
        if new_room != self.current_room:
            self.update_room_value = new_room
        else:
            self.update_room_value = None

    def update_state(self, ram, hard=False):
        self.update_global_state(ram)
        self.update_current_room(ram, hard=hard)
        self.update_agent_sector(ram, hard=hard)

    def get_abstract_state(self):
        return MRAbstractState(self.current_room, self.agent_sector if self.use_sectors else None, self.global_state)

    def abstraction_function(self, x):
        new_RAM = self.env.getRAM()
        if np.any(new_RAM != self.old_RAM):
            self.update_state(new_RAM)
            self.old_RAM = new_RAM
        return self.get_abstract_state()


class MRAbstractState(AbstractState):

    def __init__(self, room, agent_sector, global_state):
        self.room = room
        self.agent_sector = agent_sector
        self.global_state = tuple(global_state)

    def get_key_lazy(self):
        return (self.room,) + (() if self.agent_sector is None else self.agent_sector) + self.global_state

    def get_vector_lazy(self):
        onehot_room = [0] * 24
        onehot_room[self.room] = 1
        if self.agent_sector is not None:
            onehot_sector = [0] * 9
            onehot_sector[3*self.agent_sector[1] + self.agent_sector[0]] = 1
        else:
            onehot_sector = []
        posneg_global_state = [1 if x == True else -1 for x in self.global_state]
        return onehot_room + onehot_sector + posneg_global_state

    def __str__(self):
        bools = ''.join(str(int(b)) for b in self.global_state)
        if self.agent_sector is not None:
            return '%s - %s - %s' % (str(self.room), str(self.agent_sector), bools)
        else:
            return '%s - %s' % (str(self.room), bools)
