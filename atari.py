from ale_python_interface import ALEInterface
import interfaces
import numpy as np
import cv2

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
        self.last_two_frames = [np.zeros((h, w), dtype=np.float32), np.zeros((h, w), np.float32)]


    def _get_frame(self):
        image = np.zeros(self.screen_height*self.screen_width*3, dtype=np.uint8)
        self.ale.getScreenRGB(image)
        image = image.reshape([self.screen_height, self.screen_width, 3])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        image = cv2.resize(image, (84, 84))
        return image

    def perform_action(self, action):
        state = self.get_current_state()
        reward = 0
        for i in range(self.frame_skip):
            reward += self.ale.act(action)
            if i >= self.frame_skip - 2:
                self.last_two_frames = [self.last_two_frames[1], self._get_frame()]
        next_state = self.get_current_state()
        is_terminal = self.is_current_state_terminal()
        return state, action, reward, next_state, is_terminal



    def get_current_state(self):
        return np.max(self.last_two_frames, axis=0)


    def get_actions_for_state(self, state):
        return self.ale.getMinimalActionSet()

    def reset_environment(self):
        self.ale.reset_game()

    def is_current_state_terminal(self):
        return self.ale.game_over()

