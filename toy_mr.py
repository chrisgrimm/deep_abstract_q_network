import cv2
import datetime
import pygame
from interfaces import Environment
import numpy as np
import os
from random import choice
from embedding_dqn.rmax_learner import AbstractState

GRID_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)

AGENT_COLOR = (100, 100, 100)
WALL_COLOR = (0, 0, 0)
KEY_COLOR = (218,165,32)
DOOR_COLOR = (50, 50, 255)
TRAP_COLOR = (255, 0, 0)
LIVES_COLOR = (0, 255, 0)


# ACTIONS
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

# Cell Code
WALL_CODE = 1
KEY_CODE = 2
DOOR_CODE = 3
TRAP_CODE = 4
AGENT_CODE = 5
LIVES_CODE = 5

class Room():

    # TODO: allow multiple doors/keys in a single room?
    def __init__(self, loc, room_size):
        self.loc = loc
        self.size = room_size
        self.map = np.zeros(room_size, dtype=np.uint8)

    def generate_lists(self):
        self.walls = set()
        self.keys = set()
        self.doors = set()
        self.traps = set()

        for x in xrange(self.size[0]):
            for y in xrange(self.size[1]):
                if self.map[x, y] != 0:
                    if self.map[x, y] == WALL_CODE:
                        self.walls.add((x, y))
                    elif self.map[x, y] == KEY_CODE:
                        self.keys.add((x, y))
                    elif self.map[x, y] == DOOR_CODE:
                        self.doors.add((x, y))
                    elif self.map[x, y] == TRAP_CODE:
                        self.traps.add((x, y))

    def reset(self):
        self.generate_lists()


room_mapping = [(5,1), (4,1), (6,1), (3,2), (4,2), (5,2), (6,2),
                (7,2), (2,3), (3,3), (4,3), (5,3), (6,3), (7,3),
                (8,3), (1,4), (2,4), (3,4), (4,4), (5,4), (6,4),
                (7,4), (8,4), (9,4)]
room_mapping = dict([(x, i) for i, x in enumerate(room_mapping)])


class ToyMRAbstractState(AbstractState):

    def __init__(self, room_tuple, sector_num, key_states, door_states):
        self.room_tuple = room_tuple
        self.sector_num = sector_num
        self.key_states = tuple(key_states)
        self.door_states = tuple(door_states)

    def get_key_lazy(self):
        return self.room_tuple + (self.sector_num,) + self.key_states + self.door_states

    def get_vector_lazy(self):
        onehot_room = [0] * len(room_mapping)
        onehot_room[room_mapping[self.room_tuple]] = 1
        maxlen_sector = 10
        onehot_sector = [0] * maxlen_sector
        onehot_sector[self.sector_num] = 1
        posneg_key_states = [1 if x == True else -1 for x in self.key_states]
        posneg_door_states = [1 if x == True else -1 for x in self.door_states]
        return onehot_room + onehot_sector + posneg_key_states + posneg_door_states




class ToyMR(Environment):

    def __init__(self, map_file, abstraction_file=None, max_num_actions=10000, max_lives=1, repeat_action_probability=0.0, use_gui=True):

        self.rooms, self.starting_room, self.starting_cell, self.goal_room, self.keys, self.doors = self.parse_map_file(map_file)
        if abstraction_file is not None:
            self.rooms_abs, self.rooms_abs_numeric_map = self.parse_abs_file(abstraction_file)
        else:
            self.rooms_abs, self.rooms_abs_numeric_map = None, None
        self.room = self.starting_room
        self.agent = self.starting_cell
        self.num_keys = 0
        self.max_lives = max_lives
        self.lives = max_lives
        self.enter_cell = self.agent
        self.repeat_action_probability = repeat_action_probability
        self.previous_action = 0
        self.terminal = False
        self.max_num_actions = max_num_actions
        self.discovered_rooms = set()

        self.key_neighbor_locs = []
        self.door_neighbor_locs = []
        if abstraction_file is None:
            for i, (key_room, key_pos) in enumerate(self.keys):
                self.key_neighbor_locs.append(key_room)
            for i, (door_room, door_pos) in enumerate(self.doors):
                self.door_neighbor_locs.append(door_room)
        else:
            for i, (key_room, key_pos) in enumerate(self.keys):
                self.key_neighbor_locs.append(self.neighbor_locs(key_room, key_pos))
            for i, (door_room, door_pos) in enumerate(self.doors):
                self.door_neighbor_locs.append(self.neighbor_locs(door_room, door_pos))


        self.use_gui = use_gui

        # useful game dimensions
        self.tile_size = 10

        self.hud_height = 10

        self.action_ticker = 0

        pygame.init()

        # create screen
        if self.use_gui:
            self.screen = pygame.display.set_mode((self.room.size[0] * self.tile_size, self.room.size[1] * self.tile_size + self.hud_height))
        else:
            self.screen = pygame.Surface((self.room.size[0] * self.tile_size, self.room.size[1] * self.tile_size + self.hud_height))
        self.last_refresh = datetime.datetime.now()
        self.refresh_time = datetime.timedelta(milliseconds=1000 / 60)

        # load assets
        #self.key_image = pygame.image.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), './mr_maps/mr_key.png')).convert_alpha()
        #self.key_image = pygame.transform.scale(self.key_image, (self.hud_height, self.hud_height))

        self.create_sectors()

        self.screen.fill(BACKGROUND_COLOR)
        #self.draw()
        self.generate_new_state()

    def flood(self, (y, x), symbol, unchecked_sections, whole_room):
        height = len(whole_room)
        width = len(whole_room[0])
        flood_area = set([(y, x)])
        to_flood = set([(y, x)])
        while to_flood:
            (y, x) = next(iter(to_flood))
            unchecked_sections.remove((y, x))
            to_flood.remove((y, x))
            neighbors = [(y, x) for (y, x) in [(y + 1, x), (y - 1, x), (y, x - 1), (y, x + 1)]
                         if 0 <= x < width and 0 <= y < height and (y, x) in unchecked_sections
                         and whole_room[y][x] == symbol]
            for n in neighbors:
                to_flood.add(n)
                flood_area.add(n)
        return flood_area

    def check_room_abstraction_consistency(self, whole_room, room_number):
        height = len(whole_room)
        width = len(whole_room[0])
        unchecked_sections = set([(y, x) for x in range(width) for y in range(height)
                                  if whole_room[y][x] != '|'])
        symbol_area_mapping = dict()
        while unchecked_sections:
            (y, x) = section_to_check = next(iter(unchecked_sections))
            symbol = whole_room[y][x]
            flood_area = self.flood((y,x), symbol, unchecked_sections, whole_room)
            if symbol in symbol_area_mapping:
                raise Exception('Improper Abstraction in Room %s with symbol %s' % (room_number, symbol))
            else:
                symbol_area_mapping[symbol] = flood_area

    def parse_abs_file(self, abs_file):
        r = -1
        rooms = {}
        whole_room = []
        with open(abs_file) as f:
            lines = f.read().splitlines()
        for line in lines:
            print '-%s-' % line, 'len:', len(line)
            if r == -1:
                room_x, room_y, room_w, room_h = map(int, line.split(' '))
                room = {}
                curr_loc = (room_x, room_y)
                r = 0
            else:
                if len(line) == 0:
                    self.check_room_abstraction_consistency(whole_room, curr_loc)
                    whole_room = []
                    rooms[curr_loc] = room
                    r = -1
                elif line == 'G':
                    goal_room = room
                else:
                    whole_room.append(line)
                    for c, char in enumerate(line):
                        if char == '|':
                            # make sure this symbol is a wall
                            if (c, r) not in self.rooms[curr_loc].walls:
                                raise Exception('No wall at \'(%s, %s)\' location' % (curr_loc, (c, r)))
                        else:
                            room[c, r] = char
                    r += 1
        if r >= 0:
            rooms[curr_loc] = room

        # construct numeric representation of each sector
        rooms_numeric_repr = {}
        for room_xy, room_sector_dict in rooms.items():
            numeric_repr = {val: i for i, val in enumerate(set(room_sector_dict.values()))}
            rooms_numeric_repr[room_xy] = numeric_repr
        # rooms contains a mapping from coordinates to abstraction symbols (can be characters or numbers)
        # rooms_numeric_repr contains a convenience mapping from symbols to integers. (useful in case you need more
        # than 10 abstract states for a room.
        return rooms, rooms_numeric_repr

    def parse_map_file(self, map_file):
        rooms = {}
        keys = {}
        doors = {}

        r = -1
        starting_room, starting_cell, goal_room = None, None, None
        with open(map_file) as f:
            for line in f.read().splitlines():
                if r == -1:
                    room_x, room_y, room_w, room_h = map(int, line.split(' '))
                    room = Room((room_x, room_y), (room_w, room_h))
                    r = 0
                else:
                    if len(line) == 0:
                        room.generate_lists()
                        rooms[room.loc] = room
                        r = -1
                    elif line == 'G':
                        goal_room = room
                    else:
                        for c, char in enumerate(line):
                            if char == '1':
                                room.map[c, r] = '1'
                            elif char == 'K':
                                room.map[c, r] = KEY_CODE
                                keys[(room.loc, (c, r))] = True
                            elif char == 'D':
                                room.map[c, r] = DOOR_CODE
                                doors[(room.loc, (c, r))] = True
                            elif char == 'T':
                                room.map[c, r] = TRAP_CODE
                            elif char == 'S':
                                starting_room = room
                                starting_cell = (c, r)
                        r += 1
        if r >= 0:
            room.generate_lists()
            rooms[room.loc] = room

        if starting_room is None or starting_cell is None:
            raise Exception('You must specify a starting location and goal room')
        return rooms, starting_room, starting_cell, goal_room, keys, doors

    def _get_delta(self, action):
        dx = 0
        dy = 0
        if action == NORTH:
            dy = -1
        elif action == SOUTH:
            dy = 1
        elif action == EAST:
            dx = 1
        elif action == WEST:
            dx = -1
        return dx, dy

    def _move_agent(self, action):
        dx, dy = self._get_delta(action)
        return (self.agent[0] + dx, self.agent[1] + dy)

    def perform_action(self, action):
        if self.repeat_action_probability > 0:
            if np.random.uniform() < self.repeat_action_probability:
                action = self.previous_action
            self.previous_action = action

        start_state = self.get_current_state()

        new_agent = self._move_agent(action)
        reward = 0

        # room transition checks
        if (new_agent[0] < 0 or new_agent[0] >= self.room.size[0] or
            new_agent[1] < 0 or new_agent[1] >= self.room.size[1]):
            room_dx = 0
            room_dy = 0

            if new_agent[0] < 0:
                room_dx = -1
                new_agent = (self.room.size[0] - 1, new_agent[1])
            elif new_agent[0] >= self.room.size[0]:
                room_dx = 1
                new_agent = (0, new_agent[1])
            elif new_agent[1] < 0:
                room_dy = -1
                new_agent = (new_agent[0], self.room.size[1] - 1)
            elif new_agent[1] >= self.room.size[1]:
                room_dy = 1
                new_agent = (new_agent[0], 0)

            new_room = self.rooms[(self.room.loc[0] + room_dx, self.room.loc[1] + room_dy)]

            # check intersecting with adjacent door
            if new_agent in new_room.doors:
                if self.num_keys > 0:
                    new_room.doors.remove(new_agent)
                    self.num_keys -= 1
                    self.doors[(new_room.loc, new_agent)] = False

                    self.room = new_room
                    self.agent = new_agent
                    self.enter_cell = new_agent
            else:
                self.room = new_room
                self.agent = new_agent
                self.enter_cell = new_agent

            if self.room == self.goal_room:
                reward = 1
                self.terminal = True
        else:
            # collision checks
            if new_agent in self.room.keys:
                cell_type = KEY_CODE
            elif new_agent in self.room.doors:
                cell_type = DOOR_CODE
            elif new_agent in self.room.walls:
                cell_type = WALL_CODE
            elif new_agent in self.room.traps:
                cell_type = TRAP_CODE
            else:
                cell_type = 0

            if cell_type == 0:
                self.agent = new_agent
            elif cell_type == KEY_CODE:
                if self.keys[(self.room.loc, new_agent)]:
                    assert new_agent in self.room.keys
                    self.room.keys.remove(new_agent)
                    self.num_keys += 1

                    assert (self.room.loc, new_agent) in self.keys
                    self.keys[(self.room.loc, new_agent)] = False
                self.agent = new_agent
            elif cell_type == DOOR_CODE:
                if self.num_keys > 0:
                    assert new_agent in self.room.doors
                    self.room.doors.remove(new_agent)
                    self.num_keys -= 1
                    self.agent = new_agent

                    assert (self.room.loc, new_agent) in self.doors
                    self.doors[(self.room.loc, new_agent)] = False
            elif cell_type == TRAP_CODE:
                self.lives -= 1
                if self.lives == 0:
                    self.terminal = True
                else:
                    self.agent = self.enter_cell

        self.action_ticker += 1

        if self.use_gui:
            self.refresh_gui()
        
        self.generate_new_state()

        self.discovered_rooms.add(self.room.loc)
        return start_state, action, reward, self.get_current_state(), self.is_current_state_terminal()

    def create_sectors(self):
        self.sectors = dict()
        self.sectors[(5, 1)] = []

        self.sectors[(5, 1)].append(self.create_rect_set(0,0,11,3))
        self.sectors[(5, 1)].append(self.create_rect_set(4,3,3,5))
        self.sectors[(5, 1)].append(self.create_rect_set(7,3,4,8))
        self.sectors[(5, 1)].append(self.create_rect_set(4,8,3,3))
        self.sectors[(5, 1)].append(self.create_rect_set(0,3,4,8))

    def create_rect_set(self, min_x, min_y, w, h):
        rect = set()
        for x in range(min_x, min_x + w):
            for y in range(min_y, min_y + h):
                rect.add((x, y))
        return rect

    def generate_new_state(self):
        self.render_screen()
        image = pygame.surfarray.array3d(self.screen)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.state = cv2.resize(image, (84, 84))

    def get_current_state(self):
        return [self.state]

    def sector_for_loc(self, room_loc, pos):
        return self.rooms_abs_numeric_map[room_loc][self.rooms_abs[room_loc][pos]]

    def sector_abstraction(self, state):
        if self.rooms_abs is None:
            raise Exception('Cant use sector abstraction if no abstraction file is provided to ToyMR constructor.')
        sector = self.rooms_abs_numeric_map[self.room.loc][self.rooms_abs[self.room.loc][self.agent]]
        #return self.room.loc + (sector,) + tuple(np.array(self.keys.values(), dtype=int)) + tuple(np.array(self.doors.values(), dtype=int))
        return ToyMRAbstractState(self.room.loc, sector, self.keys.values(), self.doors.values())

    def oo_sector_abstraction(self, state):
        if self.rooms_abs is None:
            raise Exception('Cant use sector abstraction if no abstraction file is provided to ToyMR constructor.')
        sector = self.sector_for_loc(self.room.loc, self.agent)

        # create attributes
        attrs = dict()
        loc_att = (self.room.loc, sector)
        attrs['.loc'] = loc_att
        attrs['.num_keys'] = self.num_keys

        for i, val in enumerate(self.keys.values()):
            attrs['key_%s' % i] = val

        for i, val in enumerate(self.doors.values()):
            attrs['door_%s' % i] = val

        return tuple(sorted(attrs.items()))

    def oo_abstraction(self, state):
        # create attributes
        attrs = dict()
        loc_att = self.room.loc
        attrs['.loc'] = loc_att
        attrs['.num_keys'] = self.num_keys

        for i, val in enumerate(self.keys.values()):
            attrs['key_%s' % i] = val

        for i, val in enumerate(self.doors.values()):
            attrs['door_%s' % i] = val

        return tuple(sorted(attrs.items()))

    def predicate_func(self, l1_state):

        s = dict(l1_state)
        room = s['.loc']
        num_keys = s['.num_keys']

        # create predicates
        preds = dict()
        for i, key_neighbors in enumerate(self.key_neighbor_locs):
            pred = room == key_neighbors and s['key_%s' % i]
            preds['key_%s_in' % i] = pred
        for i, door_neighbors in enumerate(self.door_neighbor_locs):
            pred = room == door_neighbors and s['door_%s' % i]
            pred_key_door = pred and num_keys >= 1
            preds['door_%s_in' % i] = pred
            preds['door_key_%s_in' % i] = pred_key_door

        return tuple(sorted(preds.items()))

    def sector_predicate_func(self, l1_state):

        s = dict(l1_state)
        (room, sector) = s['.loc']
        num_keys = s['.num_keys']

        # create predicates
        preds = dict()
        for i, key_neighbors in enumerate(self.key_neighbor_locs):
            pred = (room, sector) in key_neighbors and s['key_%s' % i]
            preds['key_%s_in' % i] = pred
        for i, door_neighbors in enumerate(self.door_neighbor_locs):
            pred = (room, sector) in door_neighbors and s['door_%s' % i]
            pred_key_door = pred and num_keys >= 1
            preds['door_%s_in' % i] = pred
            preds['door_key_%s_in' % i] = pred_key_door

        return tuple(sorted(preds.items()))

    def neighbor_locs(self, room, loc):
        neighbors = set()
        for a in [0, 1, 2, 3]:
            dx, dy = self._get_delta(a)
            new_loc = (loc[0] + dx, loc[1] + dy)

            # room transition checks
            if (new_loc[0] < 0 or new_loc[0] >= self.room.size[0] or
                new_loc[1] < 0 or new_loc[1] >= self.room.size[1]):
                room_dx = 0
                room_dy = 0

                if new_loc[0] < 0:
                    room_dx = -1
                    new_loc = (self.room.size[0] - 1, new_loc[1])
                elif new_loc[0] >= self.room.size[0]:
                    room_dx = 1
                    new_loc = (0, new_loc[1])
                elif new_loc[1] < 0:
                    room_dy = -1
                    new_loc = (new_loc[0], self.room.size[1] - 1)
                elif new_loc[1] >= self.room.size[1]:
                    room_dy = 1
                    new_loc = (new_loc[0], 0)

                new_room = room[0] + room_dx, room[1] + room_dy
            else:
                new_room = room

            if new_loc in self.rooms[new_room].walls:
                continue

            neighbor_loc = (new_room, self.sector_for_loc(new_room, new_loc))
            neighbors.add(neighbor_loc)

        return neighbors

    #def sector_abstraction(self, state):
    #    sector = -1
    #    if self.room.loc in self.sectors:
    #        sectors = self.sectors[self.room.loc]
    #
    #        for i, sector_set in enumerate(sectors):
    #            if self.agent in sector_set:
    #                sector = i
    #                break
    #        assert sector != -1
    #    else:
    #        sector = 0
    #
    #    return self.room.loc + (sector,) + tuple(np.array(self.keys.values(), dtype=int)) + tuple(np.array(self.doors.values(), dtype=int))

    def abstraction(self, state):
        return self.room.loc + tuple(self.keys.values()) + tuple(self.doors.values())

    def get_actions_for_state(self, state):
        return NORTH, EAST, SOUTH, WEST

    def reset_environment(self):
        self.room = self.starting_room
        self.agent = self.starting_cell
        self.num_keys = 0
        self.terminal = False
        self.action_ticker = 0
        self.lives = self.max_lives
        self.enter_cell = self.agent

        for room in self.rooms.values():
            room.reset()

        for key, val in self.keys.iteritems():
            self.keys[key] = True

        for key, val in self.doors.iteritems():
            self.doors[key] = True

        if self.use_gui:
            pygame.display.update()

        self.generate_new_state()

    def is_current_state_terminal(self):
        return self.terminal or self.action_ticker > self.max_num_actions

    def is_action_safe(self, action):
        new_agent = self._move_agent(action)
        if new_agent in self.room.traps:
            return False
        return True

    def get_discovered_rooms(self):
        return self.discovered_rooms

    def render_screen(self):
        # clear screen
        self.screen.fill(BACKGROUND_COLOR)

        # loop through each row
        for row in range(self.room.size[1] + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (0, row * self.tile_size + self.hud_height),
                             (self.room.size[1] * self.tile_size, row * self.tile_size + self.hud_height))
        for column in range(self.room.size[0] + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (column * self.tile_size, self.hud_height),
                             (column * self.tile_size, self.room.size[0] * self.tile_size + self.hud_height))

        self.draw_circle(self.agent, AGENT_COLOR)

        # draw walls
        for coord in self.room.walls:
            self.draw_rect(coord, WALL_COLOR)

        # draw key
        for coord in self.room.keys:
            self.draw_rect(coord, KEY_COLOR)

        # draw doors
        for coord in self.room.doors:
            self.draw_rect(coord, DOOR_COLOR)

        # draw traps
        for coord in self.room.traps:
            self.draw_rect(coord, TRAP_COLOR)

        # draw hud
        for i in range(self.num_keys):
            #self.screen.blit(self.key_image, (i * (self.hud_height + 2), 0))
            self.draw_rect((i, -1), KEY_COLOR)
        if self.max_lives > 1:
            for i in range(self.lives):
                self.draw_rect((self.room.size[0] - 1 - i, -1), LIVES_COLOR)

    def render_screen_generated(self, name, walls, keys, doors, traps, agents):
        # clear screen
        self.screen.fill(BACKGROUND_COLOR)

        # loop through each row
        for row in range(self.room.size[1] + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (0, row * self.tile_size + self.hud_height),
                             (self.room.size[1] * self.tile_size, row * self.tile_size + self.hud_height))
        for column in range(self.room.size[0] + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (column * self.tile_size, self.hud_height),
                             (column * self.tile_size, self.room.size[0] * self.tile_size + self.hud_height))

        # draw walls
        for coord in walls:
            self.draw_rect(coord, WALL_COLOR)

        # draw key
        for coord in keys:
            self.draw_rect(coord, KEY_COLOR)

        # draw doors
        for coord in doors:
            self.draw_rect(coord, DOOR_COLOR)

        # draw traps
        for coord in traps:
            self.draw_rect(coord, TRAP_COLOR)

        # draw traps
        for coord in agents:
            self.draw_rect(coord, AGENT_COLOR)

        pygame.image.save(self.screen, './' + name + '.png')

    def draw_probability_screen(self, name, positions, intensities):
        # clear screen
        self.screen.fill(BACKGROUND_COLOR)

        # loop through each row
        for row in range(self.room.size[1] + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (0, row * self.tile_size + self.hud_height),
                             (self.room.size[1] * self.tile_size, row * self.tile_size + self.hud_height))
        for column in range(self.room.size[0] + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (column * self.tile_size, self.hud_height),
                             (column * self.tile_size, self.room.size[0] * self.tile_size + self.hud_height))

        for position, intensity in zip(positions, intensities):
            self.draw_rect(position, intensity)
        #self.draw_circle(self.agent, AGENT_COLOR)

        # draw walls
        for coord in self.room.walls:
            self.draw_rect(coord, WALL_COLOR)

        # draw key
        for coord in self.room.keys:
            self.draw_rect(coord, WALL_COLOR)

        # draw doors
        for coord in self.room.doors:
            self.draw_rect(coord, WALL_COLOR)

        # draw traps
        for coord in self.room.traps:
            self.draw_rect(coord, WALL_COLOR)

        # draw hud
        for i in range(self.num_keys):
            self.screen.blit(self.key_image, (i * (self.hud_height + 2), 0))

        pygame.image.save(self.screen, './'+name+'.png')

    def draw(self):
        self.render_screen()

        # update the display
        if self.use_gui == True:
            pygame.display.update()

    def refresh_gui(self):
        current_time = datetime.datetime.now()
        if (current_time - self.last_refresh) > self.refresh_time:
            self.last_refresh = current_time
            self.draw()

    def draw_circle(self, coord, color):
        rect = (coord[0] * self.tile_size, coord[1] * self.tile_size + self.hud_height, self.tile_size, self.tile_size)
        pygame.draw.ellipse(self.screen, color, rect)

    def draw_rect(self, coord, color):
        rect = (coord[0] * self.tile_size, coord[1] * self.tile_size + self.hud_height, self.tile_size, self.tile_size)
        pygame.draw.rect(self.screen, color, rect)

    def save_map(self, file_name, draw_sectors=False):

        map_h = 4
        map_w = 9

        map = pygame.Surface((self.tile_size * self.room.size[0] * map_w, self.tile_size * self.room.size[1] * map_h))
        map.fill(BACKGROUND_COLOR)

        for room_loc in self.rooms:
            room = self.rooms[room_loc]

            room_x, room_y = room_loc
            room_x = (room_x - 1) * self.tile_size * room.size[0]
            room_y = (room_y - 1) * self.tile_size * room.size[1]

            if room == self.goal_room:
                rect = (room_x, room_y, self.tile_size * room.size[0], self.tile_size * room.size[1])
                pygame.draw.rect(map, (0, 255, 255), rect)

                myfont = pygame.font.SysFont('Helvetica', 85)

                # render text
                label = myfont.render("G", True, (0, 0, 0))
                label_rect = label.get_rect(center=(room_x + (self.tile_size * room.size[0])/2, room_y + (self.tile_size * room.size[1])/2))
                map.blit(label, label_rect)
                continue

            # draw sectors
            if draw_sectors:
                tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                             (44, 160, 44), (152, 223, 138), (255, 152, 150),
                             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
                for row in range(self.room.size[1]):
                    for column in range(self.room.size[0]):
                        pos = (row, column)
                        if pos not in self.rooms_abs[room_loc]:
                            continue
                        sector = self.sector_for_loc(room_loc, pos)
                        color = tableau20[2 * sector + ((room_loc[0] + room_loc[1]) % 2 == 0)]
                        rect = (row * self.tile_size + room_x, column * self.tile_size + room_y, self.tile_size, self.tile_size)
                        pygame.draw.rect(map, color, rect)

            # loop through each row
            for row in range(self.room.size[1] + 1):
                pygame.draw.line(map, GRID_COLOR, (room_x, row * self.tile_size + room_y),
                                 (room.size[1] * self.tile_size + room_x, row * self.tile_size + room_y))
            for column in range(self.room.size[0] + 1):
                pygame.draw.line(map, GRID_COLOR, (column * self.tile_size + room_x, room_y),
                                 (column * self.tile_size + room_x, room.size[0] * self.tile_size + room_y))

            # draw walls
            for coord in room.walls:
                rect = (coord[0] * self.tile_size + room_x, coord[1] * self.tile_size + room_y, self.tile_size, self.tile_size)
                pygame.draw.rect(map, WALL_COLOR, rect)

            # draw key
            for coord in room.keys:
                rect = (coord[0] * self.tile_size + room_x, coord[1] * self.tile_size + room_y, self.tile_size, self.tile_size)
                pygame.draw.rect(map, KEY_COLOR, rect)

            # draw doors
            for coord in room.doors:
                rect = (coord[0] * self.tile_size + room_x, coord[1] * self.tile_size + room_y, self.tile_size, self.tile_size)
                pygame.draw.rect(map, DOOR_COLOR, rect)

            # draw traps
            for coord in room.traps:
                rect = (coord[0] * self.tile_size + room_x, coord[1] * self.tile_size + room_y, self.tile_size, self.tile_size)
                pygame.draw.rect(map, TRAP_COLOR, rect)

        pygame.image.save(map, './'+file_name+'.png')


if __name__ == "__main__":
    map_file = 'mr_maps/four_rooms.txt' # 'mr_maps/full_mr_map.txt'
    abs_file = None # 'mr_maps/full_mr_map_abs.txt'
    game = ToyMR(map_file, abstraction_file=abs_file, use_gui=True, max_lives=5)

    # map_image_file = 'mr_maps/full_mr_map'
    # game.save_map(map_image_file)
    # map_image_file = 'mr_maps/full_mr_map_sectors'
    # game.save_map(map_image_file, draw_sectors=True)
    #
    # pygame.image.save(game.screen, './' + 'example_input' + '.png')

    abs_func = game.oo_abstraction
    pred_func = game.predicate_func
    l1_state = abs_func(None)
    print l1_state, pred_func(l1_state)

    running = True
    while running:
        game.refresh_gui()

        # respond to human input
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = NORTH
                elif event.key == pygame.K_DOWN:
                    action = SOUTH
                elif event.key == pygame.K_RIGHT:
                    action = EAST
                elif event.key == pygame.K_LEFT:
                    action = WEST
                else:
                    action = -1

                if action != -1:
                    game.perform_action(action)

                    new_l1_state = abs_func(None)
                    if new_l1_state != l1_state:
                        l1_state = new_l1_state
                        print l1_state, pred_func(l1_state)

                    if game.is_current_state_terminal():
                        game.reset_environment()


