import atari
import pygame
import datetime

game_dir = './roms'
game = 'montezuma_revenge'

bitKeysMap = [
        0, 1, 2, 10, 3, 11, 6, 14, 4, 12, 7, 15, -1, -1, -1, -1,
        5, 13, -1, -1, 8, 16, -1, -1, 9, 17, -1, -1, -1, -1, -1, -1
]


if __name__ == "__main__":
    # create Atari environment
    env = atari.AtariEnvironment(game_dir + '/' + game + '.bin')
    num_actions = len(env.ale.getMinimalActionSet())

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
            env._act(action, 1)
            env.refresh_gui()

