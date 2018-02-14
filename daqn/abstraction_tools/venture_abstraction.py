
HALLWAY_ROOMS = [8, 9]

hallway_door_sectors = {
    0: [(2, 2), (3, 2)],
    1: [(1, 3), (0, 1)],
    2: [(2, 0), (3, 0)],
    3: [(0, 0), (1, 0)],
}

sub_room_door_sectors = {
    0: [(1, 0), (2, 0)],
    1: [(2, 2), (1, 0)],
    2: [(2, 0), (0, 0)],
    3: [(0, 1), (2, 1)],
}

item_sectors = {
    0: (0, 2),
    1: (0, 2),
    2: (2, 2),
    3: (1, 1)
}

x_min = 1.
y_min = 0.
x_max = 160. + 1.
y_max = 78. + 1.

STARTING_ROOM = 8
STARTING_SECTOR = (1, 3)


def get_bit(a, i):
    return a & (2**i) != 0


class VentureAbstraction(object):

    def __init__(self, environment, use_sectors=False):
        self.env = environment
        self.use_sectors = use_sectors

        self.current_room = STARTING_ROOM
        self.sector = STARTING_SECTOR
        self.is_in_sub_room = False
        self.item_collected = False
        self.player_died = False
        self.rooms_locked = [False] * 4

    def reset(self):
        self.current_room = STARTING_ROOM
        self.sector = STARTING_SECTOR
        self.is_in_sub_room = False
        self.item_collected = False
        self.player_died = False

    def update_sector(self, ram):
        if self.use_sectors:
            if self.current_room in HALLWAY_ROOMS:
                x_abs, y_abs = ram[85], ram[26]
                x = (x_abs - x_min) / (x_max - x_min)
                y = (y_abs - y_min) / (y_max - y_min)
                self.sector = (int(x * 4), int(y * 4))
            else:
                x_abs, y_abs = ram[79], ram[20]
                x = (x_abs - x_min) / (x_max - x_min)
                y = (y_abs - y_min) / (y_max - y_min)
                self.sector = (int(x * 3), int(y * 3))
        else:
            self.sector = 0

    def update_state(self, ram):
        # Don't transition if the game is frozen (occurs when the agent dies)

        if ram[77] == 1:
            self.player_died = True
        elif not get_bit(ram[63], 7):
            self.player_died = False

        if not self.player_died:
            self.current_room = ram[90]
            self.update_sector(ram)
            self.is_in_sub_room = self.current_room not in HALLWAY_ROOMS
            self.item_collected = self.is_in_sub_room and get_bit(ram[18], 7)

            for i in range(4):
                self.rooms_locked[i] = get_bit(ram[17], i)

    def oo_abstraction_function(self, x):
        self.update_state(self.env.getRAM())

        # create attributes
        attrs = dict()
        attrs['.loc'] = (self.current_room, self.sector)
        attrs['item_collected'] = self.item_collected

        for i, val in enumerate(self.rooms_locked):
            attrs['room_%s_locked' % i] = val

        return tuple(sorted(attrs.items()))

    def predicate_func(self, l1_state):

        s = dict(l1_state)
        room, sector = s['.loc']
        item_collected = s['item_collected']

        # create predicates
        preds = dict()
        num_rooms_locked = 0

        if room in HALLWAY_ROOMS:
            for i in hallway_door_sectors:
                room_i_locked = s['room_%s_locked' % i]
                num_rooms_locked += room_i_locked

                pred = not room_i_locked
                if self.use_sectors:
                    pred &= sector in hallway_door_sectors[i]
                preds['door_%s_without_item' % i] = pred
        else:
            for i in sub_room_door_sectors:
                pred = room == i and not item_collected
                if self.use_sectors:
                    pred &= sector in sub_room_door_sectors[i]
                preds['door_%s_without_item' % i] = pred


        for i in item_sectors:
            pred = room == i and sector == item_sectors[i]
            preds['in_item_%s' % i] = pred

        preds['all_rooms_locked'] = num_rooms_locked >= 4

        return tuple(sorted(preds.items()))