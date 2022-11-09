gp_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'G',
        'goal': 'minimize',
    },
    'parameters': {
        'detach': {
            'values': [
                {}, # no detach
                {'disc': 3}, # only disc
                {'gen': 3, 'disc': 3, 'noise': 2}, # all detach
            ],
        },
        'gp_weight': {
            'values': [0, 2, 5]
        }
    }
}

replay_buffer_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'G',
        'goal': 'minimize',
    },
    'parameters': {
        'detach': {
            'values': [
                {}, # no detach
                {'disc': 3}, # only disc
                {'gen': 3, 'disc': 3, 'noise': 2}, # all detach
            ],
        },
        'data_ratio': {
            'values': [0.2, 0.5, 0.8]
        }
    }
}

def get_sweep_config(name):
    match name:
        case 'gp':
            return gp_sweep_config
        case 'replay_buffer':
            return replay_buffer_sweep_config
        case _:
            raise ValueError('Sweep config name does not match one of the options')