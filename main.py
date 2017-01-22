import atari, dqn
import numpy as np

num_steps = 50000000
game = './pong.bin'


env = atari.AtariEnvironment('./pong.bin')
agent = dqn.DQN_Agent(len(env.ale.getMinimalActionSet()))


def evaluate_agent_reward(steps, env, agent):
    env.reset_environment()
    total_reward = 0
    episode_rewards = []
    for i in range(steps):
        if env.is_current_state_terminal():
            episode_rewards.append(total_reward)
            total_reward = 0
            env.reset_environment()
        state = env.get_current_state()
        action = agent.get_action_for_state(state)
        state, action, reward, next_state, is_terminal = env.perform_action(action)
        total_reward += reward
    if not episode_rewards:
        episode_rewards.append(total_reward)
    return episode_rewards

test_interval = 250000

step_num = 0
steps_until_test = test_interval
while step_num < num_steps:
    episode_steps, episode_reward = agent.run_learning_episode(env)
    step_num += episode_steps
    print 'Steps:', step_num, '\tEpisode Reward:', episode_reward
    steps_until_test -= episode_steps
    if steps_until_test <= 0:
        steps_until_test += test_interval
        print 'Evaluating network...'
        episode_rewards = evaluate_agent_reward(125000, env, agent)
        print 'Mean Reward:', np.mean(episode_reward)
