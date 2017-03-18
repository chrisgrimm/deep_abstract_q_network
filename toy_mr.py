import cv2
import pygame
from interfaces import Environment
import numpy as np

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

class Room():

    # TODO: allow multiple doors/keys in a single room?
    def __init__(self, loc, room_size):
        self.loc = loc
        self.size = room_size
        self.reset_map = np.zeros(room_size, dtype=np.uint8)
        self.map = self.reset_map

        self.key_collected = False
        self.door_opened = False

    def reset(self):
        self.map = self.reset_map

        self.key_collected = False
        self.door_opened = False


class ToyMR(Environment):

    def __init__(self, map_file, max_num_actions=10000):

        self.rooms, self.starting_room, self.starting_cell, self.goal_room = self.parse_map_file(map_file)
        self.room = self.starting_room
        self.agent = self.starting_cell
        self.has_key = False
        self.terminal = False
        self.max_num_actions = max_num_actions

        # useful game dimensions
        self.tile_size = 60

        self.hud_height = 60

        self.action_ticker = 0

        pygame.init()

        # create screen
        self.screen = pygame.display.set_mode((self.room.size[0] * self.tile_size, self.room.size[1] * self.tile_size + self.hud_height))

        # load assets
        self.key_image = pygame.image.load('../mr_maps/mr_key.png').convert_alpha()
        self.key_image = pygame.transform.scale(self.key_image, (self.hud_height, self.hud_height))

        self.screen.fill(BACKGROUND_COLOR)
        self.draw()
        self.generate_new_state()

    def parse_map_file(self, map_file):
        rooms = {}
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
                            elif char == 'D':
                                room.map[c, r] = DOOR_CODE
                            elif char == 'S':
                                starting_room = room
                                starting_cell = (c, r)
                        r += 1
        if r >= 0:
            rooms[room.loc] = room

        if starting_room is None or starting_cell is None:
            raise Exception('You must specify a starting location and goal room')
        return rooms, starting_room, starting_cell, goal_room

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

            self.room = self.rooms[(self.room.loc[0] + room_dx, self.room.loc[1] + room_dy)]
            self.agent = new_agent

            if self.room == self.goal_room:
                reward = 1
                self.terminal = True
        else:
            # collision checks
            cell = self.room.map[new_agent]
            if cell == 0:
                self.agent = new_agent
            elif cell == KEY_CODE:
                if not self.has_key:
                    self.room.map[new_agent] = 0
                    self.room.key_collected = True
                    self.has_key = True
                self.agent = new_agent
            elif cell == DOOR_CODE:
                if self.has_key:
                    self.room.map[new_agent] = 0
                    self.room.door_opened = True
                    self.has_key = False
                    self.agent = new_agent

        self.draw()

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
        return self.room.loc[0], self.room.loc[1], self.room.key_collected, self.room.door_opened, self.has_key

    def get_actions_for_state(self, state):
        return NORTH, EAST, SOUTH, WEST

    def reset_environment(self):
        self.room = self.starting_room
        self.agent = self.starting_cell
        self.has_key = False
        self.terminal = False
        self.action_ticker = 0

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
        for x in range(self.room.size[0]):
            for y in range(self.room.size[1]):
                if self.room.map[x, y] != 0:
                    if self.room.map[x, y] == WALL_CODE:
                        self.draw_rect((x, y), WALL_COLOR)
                    elif self.room.map[x, y] == KEY_CODE:
                        self.draw_circle((x, y), KEY_COLOR)
                    elif self.room.map[x, y] == DOOR_CODE:
                        self.draw_rect((x, y), DOOR_COLOR)

        # draw hud
        if self.has_key:
            self.screen.blit(self.key_image, (0, 0))

    def draw(self):
        self.render_screen()

        # update the display
        pygame.display.update()

    def draw_circle(self, coord, color):
        rect = (coord[0] * self.tile_size, coord[1] * self.tile_size + self.hud_height, self.tile_size, self.tile_size)
        pygame.draw.ellipse(self.screen, color, rect)

    def draw_rect(self, coord, color):
        rect = (coord[0] * self.tile_size, coord[1] * self.tile_size + self.hud_height, self.tile_size, self.tile_size)
        pygame.draw.rect(self.screen, color, rect)

if __name__ == "__main__":
    map_file = 'mr_maps/four_rooms.txt'
    game = ToyMR(map_file)

    running = True
    while running:

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

                    if game.is_current_state_terminal():
                        running = False


