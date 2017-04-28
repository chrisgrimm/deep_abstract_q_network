import datetime

import numpy as np
import pygame
import tqdm
import os

import atari
import atari_dqn
import coin_game
import dq_learner
import toy_mr
import wind_tunnel
import daqn
import tabular_dqn
import tabular_coin_game
from embedding_dqn import mr_environment
from embedding_dqn.abstraction_tools import montezumas_abstraction as ma
import l0_learner
import scipy.misc

# import daqn_clustering
# import dq_learner_priors
from embedding_dqn import rmax_learner

game_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../roms')

def record_episode(steps, env, agent, epsilon, abs_func):
    record_dir = 'recordings/1/'

    path = [[0., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., -1., 1.],
            [1., 0., -1., 1.],
            [0., 0., -1., 1.],
            [0., 0., -1., -1.],
            [0., 1., -1., -1.]]

    path_i = 0
    env.terminate_on_end_life = False
    env.reset_environment()
    total_reward = 0
    episode_rewards = []
    for i in tqdm.tqdm(range(steps)):
        state = env.get_current_state()
        scipy.misc.imsave(record_dir + str(i) + '.png', np.transpose(pygame.surfarray.array3d(env.screen), [1, 0, 2]))

        if env.is_current_state_terminal():
            episode_rewards.append(total_reward)
            total_reward = 0
            env.reset_environment()
            path_i = 0

        # if np.random.uniform(0, 1) < epsilon:
        #     action = np.random.choice(env.get_actions_for_state(state))
        # else:
        #     action = agent.get_action(state, evaluation=True)
        sigma = abs_func(env.abstraction(None))
        if sigma != path[path_i]:
            path_i += 1
            assert sigma == path[path_i]

        action = agent.get_action(state, sigma, path[path_i+1])

        state, action, reward, next_state, is_terminal = env.perform_action(action)
        total_reward += reward
    if not episode_rewards:
        episode_rewards.append(total_reward)
    return episode_rewards

def train_rmax_daqn(env, num_actions):
    results_dir = './results/rmax_daqn/%s_fixed_terminal' % game

    training_epsilon = 0.01
    test_epsilon = 0.001

    frame_history = 1
    #dqn = atari_dqn.AtariDQN(frame_history, num_actions)
    abs_vec_func = lambda state: [float(state[0]), float(state[1])] + [1.0 if state[i] else -1.0 for i in range(2, len(state))]
    abs_size = 4
    #abs_vec_func = ma.montezuma_abstraction_vector
    #abs_size = 35 + 9
    l0_agent = l0_learner.MultiHeadedDQLearner(abs_size, num_actions, 10, restore_network_file='./toy_mr_net', replay_memory_size=1000, frame_history=frame_history)

    record_episode(10000, env, l0_agent, 0.1, abs_vec_func)

def setup_toy_mr_env():
    env = toy_mr.ToyMR('../mr_maps/four_rooms.txt')
    num_actions = len(env.get_actions_for_state(None))
    return env, num_actions

def setup_mr_env():
    from embedding_dqn.abstraction_tools import montezumas_abstraction as ma
    env = mr_environment.MREnvironment(game_dir + '/' + 'montezuma_revenge' + '.bin', abstraction_tree=ma.abstraction_tree)
    ma.abstraction_tree.setEnv(env)
    num_actions = len(env.ale.getMinimalActionSet())
    return env, num_actions


game = 'mr_100000'
#train_rmax_daqn(*setup_mr_env())
# train_rmax_daqn(*setup_mr_env())
# train_double_dqn(*setup_toy_mr_env())
train_rmax_daqn(*setup_toy_mr_env())
