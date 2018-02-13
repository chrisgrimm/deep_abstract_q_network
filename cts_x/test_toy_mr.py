import numpy as np
import pygame
import tqdm

import toy_mr
import toy_mr_encoder
import pc_cts

map_file = '../mr_maps/full_mr_map.txt'
game = toy_mr.ToyMR(map_file)

print 'Collecting data...'
states = []
for i in range(0, 10000):
    if game.is_current_state_terminal():
        game.reset_environment()
    states.append(toy_mr_encoder.encode_toy_mr_state(game))
    action = np.random.choice([0,1,2,3])
    game.perform_action(action)

print 'Training...'
cts = pc_cts.LocationDependentDensityModel((11, 11), lambda x, y: x, pc_cts.L_shaped_context)
for i in tqdm.trange(1000):
    ii = np.random.randint(0, len(states))
    cts.update(states[ii])

print 'Ready!'
game.reset_environment()
running = True
while running:
    game.refresh_gui()

    # respond to human input
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
                print cts.log_prob(toy_mr_encoder.encode_toy_mr_state(game))

                if game.is_current_state_terminal():
                    game.reset_environment()