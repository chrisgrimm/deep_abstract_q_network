
JUNGLE_GROUND   = 32
UNDER_GROUND    = 86
DEAD_LOCATION   = 223

x_min = 8.
x_max = 148. + 1.

num_sectors = 3
treasure_sector = num_sectors - 1

class PitfallAbstraction(object):

    def __init__(self, environment, use_sectors=False):
        self.current_scene = 0
        self.treasure_scene = False
        self.use_sectors = use_sectors
        self.sector = 0
        self.num_treasures = 0
        self.above_ground = False

        self.env = environment

    def update_sector(self, ram):
        if self.use_sectors:
            x_abs = ram[97]
            x = (x_abs - x_min) / (x_max - x_min)
            self.sector = int(x * num_sectors)
        else:
            self.sector = 0

    def update_state(self, ram):
        self.current_scene = ram[1]
        self.update_sector(ram)

        self.treasure_scene = ram[21] == 5
        self.num_treasures = ram[113]

        y_pos = ram[105]
        if y_pos < DEAD_LOCATION:
            if self.above_ground:
                self.above_ground = not (y_pos >= UNDER_GROUND)
            else:
                self.above_ground = y_pos <= JUNGLE_GROUND

    def oo_abstraction_function(self, x):
        self.update_state(self.env.getRAM())

        # create attributes
        attrs = dict()
        attrs['.loc'] = (self.current_scene, self.above_ground, self.sector)
        attrs['treasure_scene'] = self.treasure_scene
        attrs['num_treasures'] = self.num_treasures

        return tuple(sorted(attrs.items()))

    def predicate_func(self, l1_state):

        s = dict(l1_state)
        scene, above_ground, sector = s['.loc']
        treasure_scene = s['treasure_scene']

        # create predicates
        preds = dict()

        pred = treasure_scene and above_ground
        if self.use_sectors:
            pred &= sector == treasure_sector
        preds['in_treasure_loc'] = pred

        return tuple(sorted(preds.items()))

    def reset(self):
        pass
