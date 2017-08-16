from plots import plot_data
from plots import parse_results
import numpy as np

def plot_rewards_toy_mr():
    data_1 = parse_results.parse_results_file('./data/hadooqn_toy_mr_sectors_fake.txt')['reward'].values.astype(np.float)
    data_2 = parse_results.parse_results_file('./data/cts_toy_mr_fake.txt')['reward'].values.astype(np.float)
    data_3 = parse_results.parse_results_file('./data/double_dqn_toy_mr_fake.txt')['reward'].values.astype(np.float)

    labels = ['Hadooqn', 'Intrinsic', 'Double DQN']

    plot_data.plot_data(range(0, 200), [data_1, data_2, data_3], 'Reward', 'Millions of Frames', 'Average Test Reward',
                        labels=labels, ylim=[-.1, 1.1], yticks=[0, 1], save_file='./figures/toy_mr_reward.png')


def plot_rooms_toy_mr():
    data_1 = parse_results.parse_results_file('./data/hadooqn_toy_mr_sectors_fake.txt')['rooms'].values.astype(
        np.float)
    data_2 = parse_results.parse_results_file('./data/cts_toy_mr_fake.txt')['rooms'].values.astype(np.float)
    data_3 = parse_results.parse_results_file('./data/double_dqn_toy_mr_fake.txt')['rooms'].values.astype(np.float)

    labels = ['Hadooqn', 'Intrinsic', 'Double DQN']

    plot_data.plot_data(range(0, 200), [data_1, data_2, data_3], 'Rooms Discovered', 'Millions of Frames', 'Rooms Discovered',
                        labels=labels, ylim=None, yticks=None, save_file='./figures/toy_mr_rooms.png')


plot_rewards_toy_mr()
plot_rooms_toy_mr()