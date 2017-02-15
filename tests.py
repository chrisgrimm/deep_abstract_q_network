import numpy as np
import matplotlib.pyplot as plt
import cv2
import atari
import jun_batch_helper

env = atari.AtariEnvironment('./roms/freeway.bin')
for i in range(100):
    env.perform_action(2)
atari_sample = env.get_current_state()[-1]
print atari_sample
jun_sample = jun_batch_helper.modify_image(cv2.imread('../train/0000/00000.png'))

cv2.imshow('atari_sample', cv2.resize(atari_sample, (400, 400)))
cv2.imshow('jun_sample', cv2.resize(jun_sample, (400, 400)))
cv2.waitKey()
