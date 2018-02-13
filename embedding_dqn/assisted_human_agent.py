from environments import atari
import pygame
import datetime
from embedding_dqn.abstraction_tools import montezumas_abstraction as ma
from embedding_dqn import l0_learner
import copy
import os
game_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../roms/')
game = 'montezuma_revenge'

bitKeysMap = [
        0, 1, 2, 10, 3, 11, 6, 14, 4, 12, 7, 15, -1, -1, -1, -1,
        5, 13, -1, -1, 8, 16, -1, -1, 9, 17, -1, -1, -1, -1, -1, -1
]


if __name__ == "__main__":
    abstraction_tree = ma.abstraction_tree
    # create Atari environment
    env = atari.AtariEnvironment(game_dir + '/' + game + '.bin', frame_skip=4, abstraction_tree=abstraction_tree, terminate_on_end_life=True)
    abstraction_tree.setEnv(env)
    abs_vec_func = ma.montezuma_abstraction_vector
    abs_size = 35 + 9
    num_actions = len(env.ale.getMinimalActionSet())
    dqn = l0_learner.MultiHeadedDQLearner(abs_size, len(env.get_actions_for_state(None)), 1)
    dqn.saver.restore(dqn.sess, './mr_net')
    l1_state = abstraction_tree.get_abstract_state()
    print ma.abstraction_tree.get_agent_sector()
    right = False
    left = False
    up = False
    down = False
    fire = False

    fps = 30

    last_update = datetime.datetime.now()
    update_time = datetime.timedelta(milliseconds=1000 / fps)

    running = True
    while running:

        if env.is_current_state_terminal():
            env.reset_environment()

        # respond to human input
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    up = True
                elif event.key == pygame.K_DOWN:
                    down = True
                elif event.key == pygame.K_RIGHT:
                    right = True
                elif event.key == pygame.K_LEFT:
                    left = True
                elif event.key == pygame.K_SPACE:
                    fire = True
                elif event.key == pygame.K_z:
                    (sector_x, sector_y) = eval(raw_input('Enter Sector:'))
                    current_state = abstraction_tree.get_abstract_state()
                    next_state = copy.deepcopy(current_state)
                    next_state.sector = (sector_x, sector_y)
                    dqn.run_learning_episode(env, abs_vec_func(current_state), abs_vec_func(next_state), current_state, next_state, env.abstraction)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    up = False
                elif event.key == pygame.K_DOWN:
                    down = False
                elif event.key == pygame.K_RIGHT:
                    right = False
                elif event.key == pygame.K_LEFT:
                    left = False
                elif event.key == pygame.K_SPACE:
                    fire = False

        now = datetime.datetime.now()
        if now - last_update > update_time:
            last_update = now

            bitfield = 0
            if left == right: bitfield |= 0
            elif left: bitfield |= 0x08
            elif right: bitfield |= 0x04

            if up == down: bitfield |= 0
            elif up: bitfield |= 0x02
            elif down: bitfield |= 0x10

            if fire: bitfield |= 0x01

            action = bitKeysMap[bitfield]
            env.perform_atari_action(action)

            abstraction_tree.update_state(env.get_current_state()[-1])
            new_l1_state = abstraction_tree.get_abstract_state()

            if new_l1_state != l1_state:
                l1_state = new_l1_state
                print l1_state

