import numpy as np
import cv2


def encode_state(env):
    state = env.get_current_state()
    frame = state[-1]

    frame = cv2.resize(frame, (42, 42))
    frame = np.right_shift(frame, 5)

    return frame
