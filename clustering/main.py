import network
import batch_helpers as bh
import numpy as np
import vis_helpers as vh
import toy_mr
import pygame
import matplotlib.pyplot as plt

def train_model(num_steps=-1, disp_interval=1000, save_interval=10000, network_name='cluster_net',
                max_n=500, batch_size=32, delta_diff=1, delta_max=5):
    if num_steps == -1:
        num_steps = np.inf
    i = 1
    while i < num_steps:
        s1, s2, steps = bh.get_batch(batch_size, max_n)
        [_, loss] = network.sess.run([network.train, network.loss],
                                     feed_dict={network.inp_state1: s1,
                                                network.inp_state2: s2,
                                                network.inp_step: steps,
                                                network.inp_delta_diff: delta_diff,
                                                network.inp_delta_max: delta_max})
        print i, loss
        if i % disp_interval == 0:
            vh.build_plot('recent', 10, batch_size=batch_size, max_n=max_n)
        if i % save_interval == 0:
            network.saver.save(network.sess, './'+network_name+'.ckpt')
        i += 1

def large_visualization(num_steps=10000, network_name='cluster_net', batch_size=32, max_n=500):
    network.saver.restore(network.sess, './'+network_name+'.ckpt')

    vh.build_plot('large_visualization', num_steps, batch_size=batch_size, max_n=max_n, store_points=True)



def play_game_with_visualization(network_name='cluster_net', points_name='large_visualization'):
    game = toy_mr.ToyMR('../mr_maps/four_rooms.txt')
    network.saver.restore(network.sess, './'+network_name+'.ckpt')
    scatter_points = np.load('./'+points_name+'.npy')
    f, ax = plt.subplots()
    while True:
        if game.is_current_state_terminal():
            game.reset_environment()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = toy_mr.NORTH
                elif event.key == pygame.K_DOWN:
                    action = toy_mr.SOUTH
                elif event.key == pygame.K_RIGHT:
                    action = toy_mr.EAST
                elif event.key == pygame.K_LEFT:
                    action = toy_mr.WEST
                else:
                    action = -1

                if action != -1:
                    game.perform_action(action)
                    state = game.get_current_state()[0]
                    plt.pause(1)
                    ax.cla()
                    ax.scatter(scatter_points[:, 0], scatter_points[:, 1], color='blue')
                    [e1] = network.sess.run([network.e1], feed_dict={network.inp_state1: [state]})
                    ax.scatter(e1[:, 0], e1[:, 1], color='red')
                    plt.draw()

play_game_with_visualization()

#large_visualization()

#train_model()