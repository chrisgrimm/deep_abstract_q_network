import argparse

import configparser

from experiment import Experiment
from vanilla_dqlearner import VanillaDQLearner


class VanillaDQNExperiment(Experiment, object):
    def __init__(self, config_file):
        super(VanillaDQNExperiment, self).__init__(config_file)

    def get_agent(self):
        agent = VanillaDQLearner(self.config, self.environment)
        return agent

    def prepare_for_evaluation(self):
        self.environment.terminate_on_end_life = False

    def prepare_for_training(self):
        self.environment.reset_environment()
        self.environment.terminate_on_end_life = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input configuration file')
    parser.add_argument('-c', '--config', help='Configuration file path', required=True)
    config_file = parser.parse_args().config

    config = configparser.ConfigParser()
    config.read(config_file)

    experiment = VanillaDQNExperiment(config)
    experiment.run()
