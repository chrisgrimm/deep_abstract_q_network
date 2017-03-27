import matplotlib.pyplot as plt
import numpy as np

def display_side_by_side(save_name, image, recon):
    f, axs = plt.subplots(1, 2)
    axs[0].imshow(np.tile(image, [1, 1, 3]))
    axs[1].imshow(np.tile(recon, [1, 1, 3]))
    f.savefig('./'+save_name+'.png')