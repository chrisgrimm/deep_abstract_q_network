import pandas as pd

def parse_results_file(filename, max_lines=None):

    data = {
        'step':[],
        'reward':[],
        'rooms':[],
    }

    with open(filename, 'r') as f:
        for i, l in enumerate(f.readlines()):
            if max_lines is not None and i == max_lines:
                break
            data_line = l.split(' -- ')
            _, step = data_line[0].split(': ')
            data['step'].append(step)

            _, reward = data_line[1].split(': ')
            data['reward'].append(reward)

            if len(data_line) == 4:
                # _, states = data_line[2].split(': ')
                # data['states'].append(states)
                _, rooms = data_line[3].split(': ')
                data['rooms'].append(rooms)
            else:
                _, rooms = data_line[2].split(': ')
                data['rooms'].append(rooms)

    return pd.DataFrame(data)
