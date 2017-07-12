import atari
import pygame
import datetime

from embedding_dqn import mr_environment
from embedding_dqn.abstraction_tools import montezumas_abstraction as ma
from embedding_dqn.abstraction_tools import mr_abstraction_ram as mr_abs

game_dir = './roms'
game = 'montezuma_revenge'

bitKeysMap = [
        0, 1, 2, 10, 3, 11, 6, 14, 4, 12, 7, 15, -1, -1, -1, -1,
        5, 13, -1, -1, 8, 16, -1, -1, 9, 17, -1, -1, -1, -1, -1, -1
]


if __name__ == "__main__":
    abstraction = mr_abs.MRAbstraction()

    # create Atari environment
    # env = atari.AtariEnvironment(game_dir + '/' + game + '.bin', frame_skip=1, terminate_on_end_life=True)
    env = mr_environment.MREnvironment(game_dir + '/' + game + '.bin', frame_skip=1, terminate_on_end_life=True, use_gui=True)
    env.set_abstraction(abstraction)
    num_actions = len(env.ale.getMinimalActionSet())
    abstraction.update_state(env.getRAM())
    abstraction.env = env
    l1_state = abstraction.get_abstract_state()

    right = False
    left = False
    up = False
    down = False
    fire = False

    fps = 60

    last_update = datetime.datetime.now()
    update_time = datetime.timedelta(milliseconds=1000 / fps)

    running = True
    while running:
        if env.is_current_state_terminal():
            print 'TERMINAL'
            env.reset_environment()
            abstraction.update_state(env.getRAM())
            l1_state = abstraction.get_abstract_state()
            print l1_state

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

            new_l1_state = abstraction.abstraction_function(None)

            if new_l1_state != l1_state:
                l1_state = new_l1_state
                print l1_state

            new_l1_state = abstraction.abstraction_function(None)

            if new_l1_state != l1_state:
                l1_state = new_l1_state
                print l1_state

