
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

key_locations = [  # room of all keys. ORDERED AS ABOVE
    1, 7, 8, 14
]

door_locations = [  # room of all doors. ORDERED AS ABOVE
    1, 1, 5, 5, 17, 17
]


def get_bit(a, i):
    return a & (2**i) != 0


class MRAbstraction(object):

    def __init__(self):
        self.global_state = [0] * len(global_object_locs)
        self.env = None
        self.current_room = 1
        self.agent_sector = (1, 2)
        self.num_keys = 0

    def set_env(self, env):
        self.env = env

    def update_num_keys(self, ram):
        num_keys = 0
        for i in range(1, 5):
            num_keys += int(get_bit(ram[65], i))
        self.num_keys = num_keys

    def update_global_state(self, ram):
        for i, (byte, bit) in enumerate(global_object_locs):
            self.global_state[i] = get_bit(ram[global_object_index + byte], bit)

    def update_current_room(self, ram):
        self.current_room = ram[room_index]

    def update_state(self, ram):
        self.update_global_state(ram)
        self.update_current_room(ram)
        self.update_num_keys(ram)

    def oo_abstraction_function(self, x):
        self.update_state(self.env.getRAM())

        # create attributes
        attrs = dict()
        attrs['.loc'] = self.current_room
        attrs['.num_keys'] = self.num_keys

        for i, val in enumerate(self.global_state[0:4]):
            attrs['key_%s' % i] = val

        for i, val in enumerate(self.global_state[4:12]):
            attrs['door_%s' % i] = val

        return tuple(sorted(attrs.items()))

    def predicate_func(self, l1_state):

        s = dict(l1_state)
        room = s['.loc']
        num_keys = s['.num_keys']

        # create predicates
        preds = dict()
        for i, key_room in enumerate(key_locations):
            pred = room == key_room and s['key_%s' % i]
            preds['key_%s_in' % i] = pred
        for i, door_room in enumerate(door_locations):
            pred = room == door_room and s['door_%s' % i]
            pred_key_door = pred and num_keys >= 1
            preds['door_%s_in' % i] = pred
            preds['door_key_%s_in' % i] = pred_key_door

        return tuple(sorted(preds.items()))

