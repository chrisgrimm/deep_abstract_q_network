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
NOOP = 0
RIGHT_1 = 1
RIGHT_2 = 2
LEFT = 3

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 100


class WindTunnel(Environment):

    def __init__(self, width=100.0, step=1.0, wind=0.5, max_actions=1000):
        # useful game dimensions
        self.width = width
        self.image_num = 0
        self.agent = 0.0
        self.step = step
        self.wind = wind

        self.max_actions = max_actions
        self.action_ticker = 0
        self.terminal = False

        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.screen.fill(BACKGROUND_COLOR)

        self.refresh_time = datetime.timedelta(milliseconds=1000 / 60)
        self.last_refresh = datetime.datetime.now()

        self.generate_new_state()
        self.refresh_gui()

    def perform_action(self, action):

        start_state = self.get_current_state()

        dx = 0.0
        if action == RIGHT_1:
            if (self.agent / self.width) < 2/3.0:
                dx = self.step
        elif action == RIGHT_2:
            if (self.agent / self.width) > 1/3.0:
                dx = self.step
        elif action == LEFT:
            dx = -self.step
        elif action != NOOP:
            raise Exception('Action not recognized')

        dx -= self.wind

        self.agent += dx

        reward = 0
        # collision checks
        if self.agent < 0:
            self.agent = 0
        # check if reached the end
        elif self.agent >= self.width:
            self.agent = self.width
            reward = 1
            self.terminal = True

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

    def get_actions_for_state(self, state):
        return NOOP, RIGHT_1, RIGHT_2, LEFT

    def reset_environment(self):
        self.agent = 0
        self.terminal = False
        self.action_ticker = 0
        self.generate_new_state()

    def is_current_state_terminal(self):
        return self.terminal or self.action_ticker >= self.max_actions

    def render_screen(self):
        # clear screen
        self.screen.fill(BACKGROUND_COLOR)

        self.draw_object(self.agent, AGENT_COLOR)

    def draw(self):
        self.render_screen()
        
        # update the display
        #data = pygame.image.save(self.screen, "directed_samples/%d-random.jpg" % self.image_num)
        self.image_num += 1
        pygame.display.update()

    def refresh_gui(self):
        current_time = datetime.datetime.now()
        if (current_time - self.last_refresh) > self.refresh_time:
            self.last_refresh = current_time
            self.draw()

    def draw_object(self, coord, color):
        rect = ((coord/self.width)*(WINDOW_WIDTH - WINDOW_HEIGHT), 0, WINDOW_HEIGHT, WINDOW_HEIGHT)
        pygame.draw.ellipse(self.screen, color, rect)


if __name__ == "__main__":
    game = WindTunnel()

    refresh_time = datetime.timedelta(milliseconds=1000 / 60)
    last_refresh = datetime.datetime.now()

    running = True
    action = NOOP

    while running:

        # respond to human input
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    action = RIGHT_1
                elif event.key == pygame.K_RIGHT:
                    action = RIGHT_2
                elif event.key == pygame.K_LEFT:
                    action = LEFT
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_RIGHT or event.key == pygame.K_DOWN or event.key == pygame.K_LEFT:
                    action = NOOP

        current_time = datetime.datetime.now()
        if (current_time - last_refresh) > refresh_time:
            last_refresh = current_time
            game.perform_action(action)

        if game.is_current_state_terminal():
            running = False


