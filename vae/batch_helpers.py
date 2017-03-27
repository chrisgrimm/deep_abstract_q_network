import numpy as np
import wind_tunnel

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


get_batch = lambda batch_size: get_batch_func(batch_size, game, RandomAgent())