import argparse

from daqn.oo_rmax_learner import OORMaxLearner
from daqn_learner import DAQNLearner
from experiment import Experiment


class DAQNExperiment(Experiment, object):
    def __init__(self, config_file):
        super(DAQNExperiment, self).__init__(config_file)

    def get_agent(self):
        agent = OORMaxLearner(self.config, self.environment)
        return agent

    def prepare_for_evaluation(self):
        self.environment.terminate_on_end_life = False

    def prepare_for_training(self):
        self.environment.reset_environment()
        self.environment.terminate_on_end_life = True

    def write_episode_data(self, step, mean_reward):
        if getattr(self.environment, 'get_discovered_rooms', None):
            self.results_file.write('Step: %d -- Mean reward: %.2f -- Num Rooms: %s -- Rooms: %s\n' %
                                    (step,
                                     mean_reward,
                                     len(self.environment.get_discovered_rooms()),
                                     self.environment.get_discovered_rooms()))
        else:
            self.results_file.write('Step: %d -- Mean reward: %.2f\n' % (step, mean_reward))
        self.results_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input configuration file')
    parser.add_argument('-c', '--config', help='Configuration file path', required=True)
    config_file = parser.parse_args().config

    experiment = DAQNExperiment(config_file)
    experiment.run()
