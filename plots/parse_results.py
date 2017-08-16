import pandas as pd

def parse_results_file(filename):

    data = {
        'step':[],
        'reward':[],
        'states':[],
        'rooms':[],
    }

    with open(filename, 'r') as f:
        for l in f.readlines():
            data_line = l.split(' -- ')
            _, step = data_line[0].split(': ')
            _, reward = data_line[1].split(': ')
            _, states = data_line[2].split(': ')
            _, rooms = data_line[3].split(': ')
            data['step'].append(step)
            data['reward'].append(reward)
            data['states'].append(states)
            data['rooms'].append(rooms)

    return pd.DataFrame(data)
