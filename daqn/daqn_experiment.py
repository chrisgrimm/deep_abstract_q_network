import argparse

from daqn_learner import DAQNLearner
from experiment import Experiment


class DAQNExperiment(Experiment, object):
    def __init__(self, config_file):
        super(DAQNExperiment, self).__init__(config_file)

    def get_agent(self):
        agent = DAQNLearner(self.config, self.environment)
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

    experiment = DAQNExperiment(config_file)
    experiment.run()
