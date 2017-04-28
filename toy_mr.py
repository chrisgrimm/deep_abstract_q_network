import cv2
import datetime
import pygame
from interfaces import Environment
import numpy as np
import os

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


class ToyMR(Environment):

    def __init__(self, map_file, max_num_actions=10000):

        self.rooms, self.starting_room, self.starting_cell, self.goal_room, self.keys, self.doors = self.parse_map_file(map_file)
        self.room = self.starting_room
        self.agent = self.starting_cell
        self.num_keys = 0
        self.terminal = False
        self.max_num_actions = max_num_actions

        # useful game dimensions
        self.tile_size = 10

        self.hud_height = 10

        self.action_ticker = 0

        pygame.init()

        # create screen
        self.screen = pygame.display.set_mode((self.room.size[0] * self.tile_size, self.room.size[1] * self.tile_size + self.hud_height))
        self.last_refresh = datetime.datetime.now()
        self.refresh_time = datetime.timedelta(milliseconds=1000 / 60)

        # load assets
        self.key_image = pygame.image.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), './mr_maps/mr_key.png')).convert_alpha()
        self.key_image = pygame.transform.scale(self.key_image, (self.hud_height, self.hud_height))

        self.screen.fill(BACKGROUND_COLOR)
        self.draw()
        self.generate_new_state()

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
                self.agent = new_agent

        self.refresh_gui()

        self.action_ticker += 1

        self.generate_new_state()

        return start_state, action, reward, self.get_current_state(), self.is_current_state_terminal()

    def generate_new_state(self):
        self.render_screen()
        image = pygame.surfarray.array3d(self.screen)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.state = cv2.resize(image, (84, 84))

    def get_current_state(self):
        return [self.state]

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
            self.screen.blit(self.key_image, (i * (self.hud_height + 2), 0))

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
    game = ToyMR(map_file)

    l1_state = game.abstraction(None)
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

                    new_l1_state = game.abstraction(None)
                    if new_l1_state != l1_state:
                        l1_state = new_l1_state
                        print l1_state

                    if game.is_current_state_terminal():
                        game.reset_environment()


