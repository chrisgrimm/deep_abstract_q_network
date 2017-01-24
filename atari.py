from ale_python_interface import ALEInterface
import interfaces
import numpy as np
import cv2
import datetime
import copy
import pygame

class AtariEnvironment(interfaces.Environment):

    def __init__(self, atari_rom, frame_skip=4, random_seed=123):
        self.ale = ALEInterface()
        self.ale.setInt('random_seed', random_seed)
        self.ale.setInt('frame_skip', 1)
        self.ale.loadROM(atari_rom)
        self.frame_skip = frame_skip
        w, h = self.ale.getScreenDims()
        self.screen_width = w
        self.screen_height = h
        self.last_two_frames = [np.zeros((84, 84), dtype=np.uint8), np.zeros((84, 84), dtype=np.uint8)]
        self.frame_history = [np.zeros((84, 84, 4), dtype=np.uint8) for i in range(0, 4)]
        atari_actions = self.ale.getMinimalActionSet()
        self.atari_to_onehot = dict(zip(atari_actions, range(len(atari_actions))))
        self.onehot_to_atari = dict(zip(range(len(atari_actions)), atari_actions))
        self.screen_image = np.zeros(self.screen_height * self.screen_width, dtype=np.uint8)

        self.use_gui = True
        self.original_frame = np.zeros((h, w), dtype=np.uint8)
        self.refresh_time = datetime.timedelta(milliseconds=1000 / 60)
        self.last_refresh = datetime.datetime.now()
        if (self.use_gui):
            self.gui_screen = pygame.display.set_mode((w, h))

    def _get_frame(self):
        self.ale.getScreenGrayscale(self.screen_image)
        image = self.screen_image.reshape([self.screen_height, self.screen_width, 1])
        image = image.reshape([self.screen_height, self.screen_width, 1])
        self.original_frame = image
        image = cv2.resize(image, (84, 84))
        return image

    def perform_action(self, onehot_index_action):
        action = self.onehot_to_atari[onehot_index_action]
        state = self.get_current_state()
        reward = 0

        self.ale.setInt("frame_skip", self.frame_skip-1)
        reward += self.ale.act(action)
        self.last_two_frames = [self.last_two_frames[1], self._get_frame()]
        self.ale.setInt("frame_skip", 1)
        reward += self.ale.act(action)
        self.last_two_frames = [self.last_two_frames[1], self._get_frame()]

        if self.use_gui:
            self.refresh_gui()

        self.frame_history[:-1] = self.frame_history[1:]
        self.frame_history[-1] = np.max(self.last_two_frames, axis=0)
        next_state = self.get_current_state()
        is_terminal = self.is_current_state_terminal()
        return state, onehot_index_action, reward, next_state, is_terminal

    def get_current_state(self):
        return copy.copy(self.frame_history)


    def get_actions_for_state(self, state):
        return [self.atari_to_onehot[a] for a in self.ale.getMinimalActionSet()]

    def reset_environment(self):
        self.ale.reset_game()
        self.frame_history[-1] = self._get_frame()

    def is_current_state_terminal(self):
        return self.ale.game_over()

    def refresh_gui(self):
        current_time = datetime.datetime.now()
        if (current_time - self.last_refresh) > self.refresh_time:
            self.last_refresh = current_time

            gui_image = np.tile(np.transpose(self.original_frame, axes=(1, 0, 2)), [1, 1, 3])
            pygame.surfarray.blit_array(self.gui_screen, gui_image)
            pygame.display.update()


