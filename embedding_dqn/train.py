import datetime

import numpy as np
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
from embedding_dqn.abstraction_tools import mr_abstraction_ram as mr_abs

# import daqn_clustering
# import dq_learner_priors
from embedding_dqn import rmax_learner

num_steps = 50000000
test_interval = 250000
test_frames = 12500
game_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../roms')

vis_update_interval = 10000


def evaluate_agent_reward(steps, env, agent, epsilon):
    agent.evaluation_values = agent.run_vi(agent.evaluation_values.copy(), evaluation=True)
    num_explored_states = len(agent.transition_table.states)

    env.terminate_on_end_life = False
    env.reset_environment()
    total_reward = 0
    episode_rewards = []
    for i in tqdm.tqdm(range(steps)):
        if env.is_current_state_terminal():
            episode_rewards.append(total_reward)
            total_reward = 0
            env.reset_environment()
        state = env.get_current_state()
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(env.get_actions_for_state(state))
        else:
            action = agent.get_action(state, evaluation=True)

        state, action, reward, next_state, is_terminal = env.perform_action(action)
        total_reward += reward
    if not episode_rewards:
        episode_rewards.append(total_reward)
    return episode_rewards, num_explored_states

def train(agent, env, test_epsilon, results_dir):
    # open results file
    results_fn = '%s/%s_results.txt' % (results_dir, game)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    results_file = open(results_fn, 'w')

    step_num = 0
    steps_until_test = test_interval
    steps_until_vis_update = 0
    best_eval_reward = - float('inf')
    while step_num < num_steps:
        env.reset_environment()
        env.terminate_on_end_life = True
        start_time = datetime.datetime.now()
        episode_steps, episode_reward = agent.run_learning_episode(env)
        if episode_reward > 0:
            env.terminate_on_end_life = False
        end_time = datetime.datetime.now()
        step_num += episode_steps

        print 'Steps:', step_num, '\tEpisode Reward:', episode_reward, '\tSteps/sec:', episode_steps / (
        end_time - start_time).total_seconds()

        # print 'Steps:', step_num, '\tEpisode Reward:', episode_reward, '\tSteps/sec:', episode_steps / (
        #     end_time - start_time).total_seconds(), '\tEps:', agent.epsilon

        # steps_until_test -= episode_steps
        # if steps_until_test <= 0:
        #     steps_until_test += test_interval
        #     print 'Evaluating network...'
        #     episode_rewards, num_explored_states = evaluate_agent_reward(test_frames, env, agent, test_epsilon)
        #     mean_reward = np.mean(episode_rewards)
        #
        #     if mean_reward > best_eval_reward:
        #         best_eval_reward = mean_reward
        #         agent.save_network('%s/%s_best_net.ckpt' % (results_dir, game))
        #
        #     print 'Mean Reward:', mean_reward, 'Best:', best_eval_reward
        #     results_file.write('Step: %d -- Mean reward: %.2f -- Num Explored: %s\n' % (step_num, mean_reward, num_explored_states))
        #     results_file.flush()


def train_rmax_daqn(env, num_actions):
    results_dir = './results/rmax_daqn/%s_fixed_terminal' % game

    training_epsilon = 0.01
    test_epsilon = 0.001

    #frame_history = 1
    #dqn = atari_dqn.AtariDQN(frame_history, num_actions)
    # abs_vec_func = lambda state: [float(state[0]), float(state[1])] + [1.0 if state[i] else -1.0 for i in range(2, len(state))]
    # abs_size = 10
    frame_history = 4
    abs = mr_abs.MRAbstraction()
    env.set_abstraction(abs)
    abs.set_env(env)
    abs_func = abs.abstraction_function
    abs_size = 24 + 9 + 10
    agent = rmax_learner.RMaxLearner(abs_size, env, abs_func, frame_history=frame_history)

    train(agent, env, test_epsilon, results_dir)

def sector_abs_vec_func(state):
    onehot = np.zeros(shape=5)
    onehot[state[2]] = 1
    return [float(state[0]), float(state[1])] + onehot.tolist() + [1.0 if state[i] else -1.0 for i in range(3, len(state))]

def train_rmax_daqn_sectors(env, num_actions):
    results_dir = './results/rmax_daqn/%s_mr_concat_abstract' % game

    training_epsilon = 0.01
    test_epsilon = 0.001

    # frame_history = 1
    # abs_func = env.sector_abstraction
    # abs_vec_func = sector_abs_vec_func
    # abs_size = 15

    frame_history = 4
    abs_func = env.abstraction
    abs_vec_func = ma.montezuma_abstraction_vector
    abs_size = 35 + 9
    agent = rmax_learner.RMaxLearner(abs_size, env, abs_vec_func, abs_func, frame_history=frame_history)

    train(agent, env, test_epsilon, results_dir)

def train_double_dqn(env, num_actions):
    results_dir = './results/dqn/%s' % game

    training_epsilon = 0.01
    test_epsilon = 0.001

    frame_history = 1
    dqn = atari_dqn.AtariDQN(frame_history, num_actions)
    agent = dq_learner.DQLearner(dqn, num_actions, frame_history=frame_history, epsilon_end=training_epsilon)

    train(agent, env, test_epsilon, results_dir)

def setup_atari_env():
    # create Atari environment
    env = atari.AtariEnvironment(game_dir + '/' + game + '.bin')
    num_actions = len(env.ale.getMinimalActionSet())
    return env, num_actions

def setup_coin_env():
    env = coin_game.CoinGame()
    num_actions = 4
    return env, num_actions

def setup_wind_tunnel_env():
    env = wind_tunnel.WindTunnel()
    num_actions = len(env.get_actions_for_state(None))
    return env, num_actions

def setup_tabular_env():
    env = tabular_coin_game.TabularCoinGame()
    num_actions = len(env.get_actions_for_state(None))
    return env, num_actions

def setup_toy_mr_env():
    env = toy_mr.ToyMR('../mr_maps/full_mr_map.txt', abstraction_file='../mr_maps/full_mr_map_abs.txt', use_gui=True)
    num_actions = len(env.get_actions_for_state(None))
    return env, num_actions

def setup_mr_env():
    env = mr_environment.MREnvironment(game_dir + '/' + 'montezuma_revenge' + '.bin')
    num_actions = len(env.ale.getMinimalActionSet())
    return env, num_actions


game = 'mr_100000'
#train_rmax_daqn(*setup_mr_env())
# train_rmax_daqn(*setup_mr_env())
# train_double_dqn(*setup_toy_mr_env())
train_rmax_daqn(*setup_mr_env())
# train_rmax_daqn_sectors(*setup_toy_mr_env())
