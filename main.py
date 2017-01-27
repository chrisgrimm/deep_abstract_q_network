import atari, dqn
import numpy as np
import datetime
import tqdm

num_steps = 50000000
game = './roms/breakout.bin'


env = atari.AtariEnvironment(game)
agent = dqn.DQN_Agent(len(env.ale.getMinimalActionSet()), learning_rate=0.00005)


def evaluate_agent_reward(steps, env, agent, epsilon):
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
            action = agent.get_action(state)

        state, action, reward, next_state, is_terminal = env.perform_action(action)
        total_reward += reward
    if not episode_rewards:
        episode_rewards.append(total_reward)
    return episode_rewards

test_interval = 250000
test_frames = 125000

step_num = 0
steps_until_test = test_interval
while step_num < num_steps:
    env.terminate_on_end_life = True
    start_time = datetime.datetime.now()
    episode_steps, episode_reward = agent.run_learning_episode(env)
    end_time = datetime.datetime.now()
    step_num += episode_steps
    print 'Steps:', step_num, '\tEpisode Reward:', episode_reward, '\tSteps/sec:', episode_steps / (end_time - start_time).total_seconds(), '\tEps:', agent.epsilon
    steps_until_test -= episode_steps
    if steps_until_test <= 0:
        steps_until_test += test_interval
        print 'Evaluating network...'
        episode_rewards = evaluate_agent_reward(test_frames, env, agent, 0.05)
        print 'Mean Reward:', np.mean(episode_rewards)
