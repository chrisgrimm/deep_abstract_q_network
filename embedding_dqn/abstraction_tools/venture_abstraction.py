import numpy as np
from abstraction_interfaces import AbstractState


HALLWAY_ROOMS = [8, 9]


def get_bit(a, i):
    return a & (2**i) != 0


class VentureAbstraction(object):

    def __init__(self, environment):
        self.current_room = 8
        self.is_in_sub_room = False
        self.item_collected = False
        self.rooms_locked = [False] * 4
        self.env = environment

    def update_state(self, ram):
        self.current_room = ram[90]
        self.is_in_sub_room = self.current_room not in HALLWAY_ROOMS
        self.item_collected = self.is_in_sub_room and get_bit(ram[18], 7)

        for i in range(4):
            self.rooms_locked[i] = get_bit(ram[17], i)

    def oo_abstraction_function(self, x):
        self.update_state(self.env.getRAM())

        # create attributes
        attrs = dict()
        attrs['.room'] = self.current_room
        attrs['item_collected'] = self.item_collected

        for i, val in enumerate(self.rooms_locked):
            attrs['room_%s_locked' % i] = val

        return tuple(sorted(attrs.items()))

    def predicate_func(self, l1_state):

        s = dict(l1_state)
        room = s['.room']

        # create predicates
        preds = dict()
        for i in range(len(self.rooms_locked)):
            pred = room in HALLWAY_ROOMS and s['room_%s_locked' % i]
            preds['room_%s_locked' % i] = pred

        return tuple(sorted(preds.items()))