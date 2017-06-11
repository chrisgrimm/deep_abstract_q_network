import cv2
import datetime
import pygame
from interfaces import Environment
import numpy as np
import os
from embedding_dqn.rmax_learner import AbstractState

GRID_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (255, 255,  255)

AGENT_COLOR = (100, 100,  100)
WALL_COLOR = (0, 0, 0)
KEY_COLOR = (218,165,32)
DOOR_COLOR = (50, 50, 255)
TRAP_COLOR = (255, 0, 0)

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
        posneg_key_states = [1 if x == True else -1 for x in self.key_states]
        posneg_door_states = [1 if x == True else -1 for x in self.door_states]
        return onehot_room + [self.sector_num,] + posneg_key_states + posneg_door_states


class ToyMR(Environment):

    def __init__(self, map_file, abstraction_file=None, max_num_actions=10000, use_gui=False):

        self.rooms, self.starting_room, self.starting_cell, self.goal_room, self.keys, self.doors = self.parse_map_file(map_file)
        if abstraction_file is not None:
            self.rooms_abs, self.rooms_abs_numeric_map = self.parse_abs_file(abstraction_file)
        else:
            self.rooms_abs, self.rooms_abs_numeric_map = None, None
        self.room = self.starting_room
        self.agent = self.starting_cell
        self.num_keys = 0
        self.terminal = False
        self.max_num_actions = max_num_actions

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

    def parse_abs_file(self, abs_file):
        r = -1
        rooms = {}
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
                    rooms[curr_loc] = room
                    r = -1
                elif line == 'G':
                    goal_room = room
                else:
                    for c, char in enumerate(line):
                        room[c, r] = char
                    r += 1
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
    #@profile
    def perform_action(self, action):

        start_state = self.get_current_state()

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

        new_agent = (self.agent[0] + dx, self.agent[1] + dy)
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
                    self.doors[(self.room.loc, new_agent)] = False

                    self.room = new_room
                    self.agent = new_agent
            else:
                self.room = new_room
                self.agent = new_agent

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
                self.terminal = True

        self.action_ticker += 1

        if self.use_gui:
            self.refresh_gui()
        
        self.generate_new_state()

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
    #@profile
    def generate_new_state(self):
        self.render_screen()
        image = pygame.surfarray.array3d(self.screen)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.state = cv2.resize(image, (84, 84))

    def get_current_state(self):
        return [self.state]

    def sector_abstraction(self, state):
        if self.rooms_abs is None:
            raise Exception('Cant use sector abstraction if no abstraction file is provided to ToyMR constructor.')
        sector = self.rooms_abs_numeric_map[self.room.loc][self.rooms_abs[self.room.loc][self.agent]]
        #return self.room.loc + (sector,) + tuple(np.array(self.keys.values(), dtype=int)) + tuple(np.array(self.doors.values(), dtype=int))
        return ToyMRAbstractState(self.room.loc, sector, self.keys.values(), self.doors.values())

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
            self.draw_rect((i * (self.hud_height + 2), 0), KEY_COLOR)

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

if __name__ == "__main__":
    map_file = 'mr_maps/full_mr_map.txt'
    abs_file = 'mr_maps/full_mr_map_abs.txt'
    game = ToyMR(map_file, abstraction_file=abs_file)

    l1_state = game.sector_abstraction(None)
    print l1_state

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

                    new_l1_state = game.sector_abstraction(None)
                    if new_l1_state != l1_state:
                        l1_state = new_l1_state
                        print l1_state

                    if game.is_current_state_terminal():
                        game.reset_environment()


