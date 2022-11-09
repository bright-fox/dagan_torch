import math
import os
import numpy as np
import torch


def load_data(path):
    """
    Load data from path
    """

    train_data = np.load(f'{path}/train.npz')
    val_data = np.load(f'{path}/val.npz')

    train_data = {
        'o': train_data['orig'],
        'a': train_data['aug'],
    }
    val_data = {
        'o': val_data['orig'],
        'a': val_data['aug'],
    }

    return train_data, val_data
    
def sample_data(old, new, ratio):
    """
    Sample data for iterative training iteration
    """
    data_size = new['o'].shape[0]

    num_of_new = math.floor(data_size * ratio)
    num_of_old = data_size - num_of_new

    old_indices = np.random.choice(np.arange(old['o'].shape[0]), num_of_old, replace=False)
    new_indices = np.random.choice(np.arange(new['o'].shape[0]), num_of_new, replace=False)

    return {
        'o': np.concatenate((old['o'][old_indices], new['o'][new_indices]), axis=0),
        'a': np.concatenate((old['a'][old_indices], new['a'][new_indices]), axis=0),
    }

def update_data(old, new):
    """
    Update the dataset with the new data
    """

    return {
        'o': np.concatenate((old['o'], new['o']), axis=0),
        'a': np.concatenate((old['a'], new['a']), axis=0),
    }

def save_model(model, dir_path, prefix=''):
    """
    Save the model and its state dict in the specified directory with an optional name prefix
    """

    torch.save(model, os.path.join(dir_path, f'{prefix}_model.pt'))
    torch.save(model.state_dict(), os.path.join(dir_path, f'{prefix}_state_dict.pt'))
