import datetime
import pygame
from interfaces import Environment
import numpy as np
import cv2
import copy

GRID_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (255, 255,  255)

AGENT_COLOR = (100, 100,  100)
BUTTON_COLOR = (0, 255, 0)
COIN_COLOR = (0, 255,  255)

# ACTIONS
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3


class CoinGame(Environment):

    def __init__(self, map_width=10, map_height=10, max_actions=1000, frame_history_length=1, image_states=True):
        # useful game dimensions
        self.tile_size = 10
        self.map_width = map_width
        self.map_height = map_height

        self.agent = (0, 0)
        self.button = (0, map_height-1)
        self.coin = (map_width-1, 0)
        self.coin_is_present = True

        self.max_actions = max_actions
        self.action_ticker = 0

        self.image_states = image_states

        self.zero_history_frames = [np.zeros((84, 84), dtype=np.uint8) for i in range(0, frame_history_length)]
        self.frame_history = copy.copy(self.zero_history_frames)

        pygame.init()
        self.screen = pygame.display.set_mode((self.map_width * self.tile_size, self.map_height * self.tile_size))
        self.screen.fill(BACKGROUND_COLOR)

        # self.state_vis_screen = np.zeros((self.map_height * self.tile_size, self.map_width * self.tile_size * 2, 3), dtype=np.uint8)
        # cv2.namedWindow('state_vis')
        # cv2.startWindowThread()
        # cv2.imshow('state_vis', self.state_vis_screen)
        # cv2.waitKey(1)

        self.refresh_time = datetime.timedelta(milliseconds=1000 / 60)
        self.last_refresh = datetime.datetime.now()

        self.refresh_gui()

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
        else:
            raise Exception('Action not recognized')

        new_agent = (self.agent[0] + dx, self.agent[1] + dy)

        # collision checks
        if not (new_agent[0] < 0 or new_agent[0] >= self.map_width or
                new_agent[1] < 0 or new_agent[1] >= self.map_height):
            self.agent = new_agent

        # check if pressing button
        if self.agent[0] == self.button[0] and self.agent[1] == self.button[1]:
            self.coin_is_present = True

        reward = 0
        # check if collected coin
        if self.coin_is_present and self.agent[0] == self.coin[0] and self.agent[1] == self.coin[1]:
            self.coin_is_present = False
            reward = 1

        self.refresh_gui()

        self.action_ticker += 1

        # create new state
        if self.image_states:
            self.render_screen()
            image = pygame.surfarray.array3d(self.screen)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            state = cv2.resize(image, (84, 84))
        else:
            state = np.zeros((self.map_width * self.map_height) + 1, np.float32)

            index = self.agent[1] * self.map_width + self.agent[0]
            state[index] = 1.0

            if self.coin_is_present:
                state[-1] = 1.0

        self.frame_history[:-1] = self.frame_history[1:]
        self.frame_history[-1] = state

        return start_state, action, reward, self.get_current_state(), self.is_current_state_terminal()

    def get_current_state(self):
        return copy.copy(self.frame_history)

    def get_actions_for_state(self, state):
        return NORTH, EAST, SOUTH, WEST

    def reset_environment(self):
        self.agent = (0, 0)
        self.action_ticker = 0
        self.frame_history = copy.copy(self.zero_history_frames)
        self.coin_is_present = True

    def is_current_state_terminal(self):
        return self.action_ticker >= self.max_actions

    def render_screen(self):
        # clear screen
        self.screen.fill(BACKGROUND_COLOR)

        # loop through each row
        for row in range(self.map_height + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (0, row * self.tile_size),
                             (self.map_width * self.tile_size, row * self.tile_size))
        for column in range(self.map_width + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (column * self.tile_size, 0),
                             (column * self.tile_size, self.map_height * self.tile_size))

        self.draw_object(self.agent, AGENT_COLOR)
        self.draw_object(self.button, BUTTON_COLOR)
        if self.coin_is_present:
            self.draw_object(self.coin, COIN_COLOR)

    def draw(self):
        self.render_screen()

        # update the display
        pygame.display.update()

    def refresh_gui(self):
        current_time = datetime.datetime.now()
        if (current_time - self.last_refresh) > self.refresh_time:
            self.last_refresh = current_time
            self.draw()

    def draw_object(self, coord, color):
        rect = (coord[0] * self.tile_size, coord[1] * self.tile_size, self.tile_size, self.tile_size)
        pygame.draw.ellipse(self.screen, color, rect)

    def abstraction(self):
        return np.array([0, 1]) if self.coin_is_present else np.array([1, 0])

    def visualize_l1_states(self, l1_state_probs_tf, inp_frames, inp_mask, sess):
        current_agent = self.agent
        current_coin_is_present = self.coin_is_present

        all_states = []
        for b in range(2):
            for x in range(self.map_width):
                for y in range(self.map_height):
                    self.coin_is_present = bool(b)
                    self.agent = (x, y)
                    self.render_screen()
                    image = pygame.surfarray.array3d(self.screen)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    state = cv2.resize(image, (84, 84))
                    all_states.append(state)

        all_probs = []
        for state in all_states:
            [l1_state_probs] = sess.run([l1_state_probs_tf],
                                           feed_dict={inp_frames: np.reshape(state, [1, 84, 84, 1]),
                                                      inp_mask: np.ones((1, 1), dtype=np.float32)})
            all_probs.append(l1_state_probs[0])

        # i = 0
        # for x in range(self.map_width*2):
        #     for y in range(self.map_height):
        #         probs = all_probs[i]
        #         color = (int(probs[0]*255), 0, int(probs[1]*255))
        #
        #         pt1 = (x * self.tile_size, y * self.tile_size)
        #         pt2 = (pt1[0] + self.tile_size, pt1[1] + self.tile_size)
        #         cv2.rectangle(self.state_vis_screen, pt1, pt2, color, thickness=-1)
        #
        #         i += 1
        #
        # cv2.imshow('state_vis', self.state_vis_screen)
        # cv2.waitKey(1)
        # self.agent = current_agent
        # self.coin_is_present = current_coin_is_present

        with open('all_probs.txt', 'w') as f:
            for probs in all_probs:
                f.write('%f,%f\n' % (probs[0], probs[1]))


if __name__ == "__main__":
    game = CoinGame()

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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


