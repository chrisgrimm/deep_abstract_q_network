import datetime
import shutil

import configparser
import numpy as np
import tqdm
import os

import atari

game_dir = './roms'


class Experiment:
    def __init__(self, config_file):
        # Set configuration params
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.num_training_steps = int(self.config['EXP']['NUM_TRAINING_STEPS'])
        self.test_interval = int(self.config['EXP']['TEST_INTERVAL'])
        self.test_frames = int(self.config['EXP']['TEST_FRAMES'])
        self.test_epsilon = float(self.config['EXP']['TEST_EPSILON'])
        self.test_max_episode_frames = int(self.config['EXP']['TEST_MAX_EPISODE_FRAMES'])
        self.exp_dir = self.config['EXP']['DIRECTORY']

        self.environment, self.num_actions = self.get_environment()

        if self.environment is None:
            raise Exception('Environment ' + self.config['ENV']['ID'] + ' not found')

        self.agent = self.get_agent()

        # Create experiment directory
        if os.path.isdir(self.exp_dir):
            count = 1
            exp_dir_prime = self.exp_dir + '_' + str(count)
            while os.path.isdir(exp_dir_prime):
                exp_dir_prime = self.exp_dir + '_' + str(count)
            self.exp_dir = exp_dir_prime
        os.makedirs(self.exp_dir)
        self.results_file = open(os.path.join(self.exp_dir, 'results.txt'), 'w')
        shutil.copy(config_file, os.path.join(self.exp_dir, 'config.ini'))

    def evaluate_agent_reward(self):
        self.prepare_for_evaluation()
        self.environment.reset_environment()

        total_reward = 0
        episode_rewards = []
        episode_steps = 0

        for step in tqdm.tqdm(range(self.test_frames)):
            if episode_steps >= self.test_max_episode_frames or self.environment.is_current_state_terminal():
                episode_rewards.append(total_reward)
                total_reward = 0
                episode_steps = 0
                self.environment.reset_environment()
            state = self.environment.get_current_state()
            if np.random.uniform(0, 1) < self.test_epsilon:
                action = np.random.choice(self.environment.get_actions_for_state(state))
            else:
                action = self.agent.get_action(state, self.environment, {})

            state, action, reward, next_state, is_terminal = self.environment.perform_action(action)
            total_reward += reward
            episode_steps += 1

        if not episode_rewards:
            episode_rewards.append(total_reward)

        return episode_rewards

    def run(self):
        self.environment.terminate_on_end_life = False

        step = 0
        steps_until_test = self.test_interval
        best_eval_reward = - float('inf')
        while step < self.num_training_steps:
            self.prepare_for_training()
            self.environment.reset_environment()

            start_time = datetime.datetime.now()
            episode_steps, episode_reward = self.agent.run_learning_episode(self.environment, episode_dict={})
            end_time = datetime.datetime.now()
            step += episode_steps

            print 'Steps:', step, '\tEpisode Reward:', episode_reward, '\tSteps/sec:', episode_steps / (
                    end_time - start_time).total_seconds()

            steps_until_test -= episode_steps
            if steps_until_test <= 0:
                steps_until_test += self.test_interval
                print 'Evaluating network...'
                episode_rewards = self.evaluate_agent_reward()
                mean_reward = np.mean(episode_rewards)

                if mean_reward > best_eval_reward:
                    best_eval_reward = mean_reward
                    self.agent.save_network(os.path.join(self.exp_dir, 'best_net'))

                print 'Mean Reward:', mean_reward, 'Best:', best_eval_reward

                self.write_episode_data(step, mean_reward)

    def write_episode_data(self, step, mean_reward):
        self.results_file.write('Step: %d -- Mean reward: %.2f\n' % (step, mean_reward))
        self.results_file.flush()

    def get_environment(self):
        environment_id = self.config['ENV']['ID']

        env = None
        num_actions = 0

        if environment_id == 'atari':
            env = atari.AtariEnvironment(self.config, use_gui=True)
            num_actions = len(env.ale.getMinimalActionSet())

        return env, num_actions

    def get_agent(self):
        raise NotImplemented

    def prepare_for_evaluation(self):
        pass

    def prepare_for_training(self):
        pass
