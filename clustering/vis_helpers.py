import numpy as np
import cv2
import batch_helpers as bh
import network
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm


def handlabel_examples():
    N = bh.screens.shape[0]
    while True:
        i = np.random.randint(0, N)
        screen = bh.screens[i]
        cv2.imshow('screen', cv2.resize(np.transpose(screen), (400, 400)))
        cv2.waitKey(1)

def build_plot(save_name, num_samples, batch_size=32, max_n=5, store_points=False):
    f, ax = plt.subplots()
    all_points = []
    for i in tqdm.tqdm(range(num_samples)):
        s1, s2, n = bh.get_batch(batch_size, max_n)
        [e1] = network.sess.run([network.e1], feed_dict={network.inp_state1: s1})
        all_points.append(e1)
        ax.scatter(e1[:, 0], e1[:, 1])
    if store_points:
        all_points = np.concatenate(all_points, axis=0)
        np.save('./' + save_name + '.npy', all_points)
    f.savefig('./' + save_name+ '.png')