import numpy as np
from abstraction_interfaces import AbstractState

# RAM_LOCATIONS
RAM_X = 27
RAM_Y = 31
RAM_ROOM = 28
RAM_LEVEL = 117
RAM_POWER_METER = 43

x_min = 15.
y_min = 0.
x_max = 146. + 1.
y_max = 139. + 1.

POWER_MAX = 81

STARTING_ROOM = 0
STARTING_SECTOR = (0, 1)

class HeroAbstraction(object):

    def __init__(self, environment, use_sectors=False):
        self.current_level = 0
        self.current_room = STARTING_ROOM
        self.sector = STARTING_SECTOR
        self.ran_out_of_power = True

        self.env = environment
        self.use_sectors = use_sectors

    def reset(self):
        self.current_room = STARTING_ROOM
        self.sector = STARTING_SECTOR

    def update_sector(self, ram):
        if self.use_sectors:
            x_abs, y_abs = ram[RAM_X], ram[RAM_Y]
            x = (x_abs - x_min) / (x_max - x_min)
            y = (y_abs - y_min) / (y_max - y_min)
            self.sector = (int(x * 3), int(y * 3))
        else:
            self.sector = 0

    def update_state(self, ram):
        self.current_level = ram[RAM_LEVEL]

        if ram[RAM_POWER_METER] == 0:
            self.ran_out_of_power = True
        elif ram[RAM_POWER_METER] == POWER_MAX:
            self.ran_out_of_power = False

        if not self.ran_out_of_power:
            self.current_room = ram[RAM_ROOM]
            self.update_sector(ram)

    def oo_abstraction_function(self, x):
        self.update_state(self.env.getRAM())

        # create attributes
        attrs = dict()
        attrs['.loc'] = (self.current_level, self.current_room, self.sector)

        return tuple(sorted(attrs.items()))

    def predicate_func(self, l1_state):
        return ()
