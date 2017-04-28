import numpy as np
import toy_mr


def encode_toy_mr_state(env):
    (w, h) = env.room.map.shape
    symbols = np.zeros((w, h + 1), dtype=np.uint32)

    # draw walls
    for coord in env.room.walls:
        symbols[coord] = toy_mr.WALL_CODE

    # draw key
    for coord in env.room.keys:
        symbols[coord] = toy_mr.KEY_CODE

    # draw doors
    for coord in env.room.doors:
        symbols[coord] = toy_mr.DOOR_CODE

    # draw traps
    for coord in env.room.traps:
        symbols[coord] = toy_mr.TRAP_CODE

    symbols[env.agent] = toy_mr.AGENT_CODE

    for i in range(env.num_keys):
        symbols[i, h] = 1

    return symbols