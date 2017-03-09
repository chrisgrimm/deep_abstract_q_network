
import datetime

import numpy as np
import tqdm

import atari
import dq_learner
import atari_dqn
import coin_game
import wind_tunnel
import daqn


def evaluate_agent_reward(steps, env, agent, epsilon):
    env.terminate_on_end_life = False
    env.reset_environment()
  
    for i in tqdm.tqdm(range(steps)):
        if env.is_current_state_terminal():
            env.reset_environment()
        state = env.get_current_state()
        action = np.random.choice(env.get_actions_for_state(state))
        state, action, reward, next_state, is_terminal = env.perform_action(action)
    return episode_rewards

def train_double_dqn(env, num_actions):
    results_dir = './results/double_dqn/wind_tunnel'

    training_epsilon = 0.01
    test_epsilon = 0.001

    frame_history = 1
    dqn = atari_dqn.AtariDQN(frame_history, num_actions)
    agent = dq_learner.DQLearner(dqn, num_actions, frame_history=frame_history, epsilon_end=training_epsilon)

    train(agent, env, test_epsilon, results_dir)


def train(agent, env, test_epsilon, results_dir):
    # open results file
    results_fn = '%s/%s_results.txt' % (results_dir, game)
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
        end_time = datetime.datetime.now()
        step_num += episode_steps

        # print 'Steps:', step_num, '\tEpisode Reward:', episode_reward, '\tSteps/sec:', episode_steps / (
        # end_time - start_time).total_seconds(), '\tL1Eps:', agent.epsilon, '\tL0Eps:', agent.l0_learner.epsilon

        print 'Steps:', step_num, '\tEpisode Reward:', episode_reward, '\tSteps/sec:', episode_steps / (
            end_time - start_time).total_seconds(), '\tEps:', agent.epsilon

        steps_until_test -= episode_steps
        if steps_until_test <= 0:
            steps_until_test += test_interval
            print 'Evaluating network...'
            episode_rewards = evaluate_agent_reward(test_frames, env, agent, test_epsilon)
            mean_reward = np.mean(episode_rewards)

            if mean_reward > best_eval_reward:
                best_eval_reward = mean_reward
                agent.save_network('%s/%s_best_net.ckpt' % (results_dir, game))

            print 'Mean Reward:', mean_reward, 'Best:', best_eval_reward
            results_file.write('Step: %d -- Mean reward: %.2f\n' % (step_num, mean_reward))
            results_file.flush()


def setup_wind_tunnel_env():
    env = wind_tunnel.WindTunnel()
    num_actions = len(env.get_actions_for_state(None))
    return env, num_actions