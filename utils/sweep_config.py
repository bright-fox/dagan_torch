sweep_config = {
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