from ale_python_interface import ALEInterface
import interfaces
import numpy as np
import cv2
import datetime
import copy
import pygame
import atari





class AbstractAtariEnvironment(atari.AtariEnvironment):

    def __init__(self, abstraction_tree, atari_rom, frame_skip=4, noop_max=30, terminate_on_end_life=False, random_seed=123,
                 frame_history_length=4, use_gui=True, max_num_frames=500000):
        super(AbstractAtariEnvironment, self).__init__(atari_rom, frame_skip, noop_max, terminate_on_end_life, random_seed,
                                                       frame_history_length, use_gui, max_num_frames)
        self.abstraction_tree = abstraction_tree

    def abstraction(self, state):
        # TODO implement abstraction tree.
        pass


