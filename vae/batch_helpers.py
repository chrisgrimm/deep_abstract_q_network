import numpy as np
import wind_tunnel
import toy_mr

game = wind_tunnel.WindTunnel()

class RandomAgent(object):

    def get_action(self, game):
        actions = game.get_actions_for_state(None)
        return np.random.choice(actions)

def get_batch_func(batch_size, game, agent):
    states = []
    for i in range(batch_size):
        if game.is_current_state_terminal():
            game.reset_environment()
        action = agent.get_action(game)
        game.perform_action(action)
        states.append(game.get_current_state()[0])
    return np.array(states)

def setup_toy_mr_env():
    env = toy_mr.ToyMR('../mr_maps/full_mr_map.txt')
    num_actions = len(env.get_actions_for_state(None))
    return env, num_actions

toy_mr_env, toy_mr_num_actions = setup_toy_mr_env()
toy_mr_actions = range(toy_mr_num_actions)


def encode_toy_mr_state(env):
    (w, h) = env.room.map.shape
    symbols = np.zeros((w, h + 1, 5), dtype=np.uint32)

    # draw walls
    for coord in env.room.walls:
        symbols[coord + (toy_mr.WALL_CODE-1,)] = 1

    # draw key
    for coord in env.room.keys:
        symbols[coord + (toy_mr.KEY_CODE-1,)] = 1

    # draw doors
    for coord in env.room.doors:
        symbols[coord + (toy_mr.DOOR_CODE-1,)] = 1

    # draw traps
    for coord in env.room.traps:
        symbols[coord + (toy_mr.TRAP_CODE-1,)] = 1

    symbols[env.agent + (toy_mr.AGENT_CODE - 1,)] = 1

    for i in range(env.num_keys):
        symbols[i, h, toy_mr.KEY_CODE-1] = 1

    return symbols

# @profile
def get_toy_mr_batch(batch_size):
    batch = []
    for i in range(batch_size):
        toy_mr_env.perform_action(np.random.choice(toy_mr_actions))
        batch.append(encode_toy_mr_state(toy_mr_env))
        if toy_mr_env.is_current_state_terminal():
            toy_mr_env.reset_environment()
    return batch


get_batch = lambda batch_size: get_batch_func(batch_size, game, RandomAgent())