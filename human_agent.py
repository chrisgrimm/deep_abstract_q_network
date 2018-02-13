from environments import atari
import pygame
import datetime

from embedding_dqn.abstraction_tools import mr_abstraction_ram
from embedding_dqn.abstraction_tools import venture_abstraction

game_dir = './roms'
game = 'venture'
abstraction = None # mr_abs.MRAbstraction()

bitKeysMap = [
        0, 1, 2, 10, 3, 11, 6, 14, 4, 12, 7, 15, -1, -1, -1, -1,
        5, 13, -1, -1, 8, 16, -1, -1, 9, 17, -1, -1, -1, -1, -1, -1
]


def get_bit(a, i):
    return a & (2**i) != 0

if __name__ == "__main__":
    # create Atari environment
    env = atari.AtariEnvironment(game_dir + '/' + game + '.bin', frame_skip=1, terminate_on_end_life=True, use_gui=True, max_num_frames=72000)
    abstraction = mr_abstraction_ram.MRAbstraction(env, use_sectors=True)
    abstraction = venture_abstraction.VentureAbstraction(env, use_sectors=True)
    # abstraction = pitfall_abstraction.PitfallAbstraction(env, use_sectors=True)
    # env = mr_environment.MREnvironment(game_dir + '/' + game + '.bin', frame_skip=1, terminate_on_end_life=True, use_gui=True)
    # abstraction = hero_abstraction.HeroAbstraction(env, use_sectors=True)
    num_actions = len(env.ale.getMinimalActionSet())
    if abstraction is not None:
        abstraction.update_state(env.getRAM())
        abstraction.env = env
        env.abstraction = abstraction
        l1_state = abstraction.oo_abstraction_function(None)

    right = False
    left = False
    up = False
    down = False
    fire = False

    fps = 60
    val = 1

    last_update = datetime.datetime.now()
    update_time = datetime.timedelta(milliseconds=1000 / fps)

    running = True
    while running:
        if env.is_current_state_terminal():
            print 'TERMINAL'
            env.reset_environment()
            if abstraction is not None:
                abstraction.reset()
                l1_state = abstraction.oo_abstraction_function(None)
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
            state, atari_action, reward, next_state, is_terminal = env.perform_atari_action(action)

            # print env.getRAM()[77]

            # x = 27
            # y = 31
            # room = 28
            # level = 117
            #
            # ram = env.getRAM()
            # print '%s: %s, (%s, %s)' % (ram[level], ram[room], ram[x], ram[y])

            # 46, 84, 85, 86

            # indices = [i for i, v in enumerate(env.getRAM()) if v == val]
            # print indices

            if abstraction is not None:
                new_l1_state = abstraction.oo_abstraction_function(None)
                if new_l1_state != l1_state:
                    l1_state = new_l1_state
                    print l1_state, abstraction.predicate_func(l1_state)



