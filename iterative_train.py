import pathlib
import random
import uuid
import wandb
from dagan_trainer import DaganTrainer
from dagan_torch.discriminator import Discriminator
from dagan_torch.generator import Generator
from dagan_torch.dataset import create_dl
from utils.parser import get_dagan_args, prepare_args
from utils.utils import create_output_dirs, load_data, sample_data, save_model, update_data
from utils.logger import Logger
from utils.sweep_config import sweep_config
import torch
import os
import torch.optim as optim
import numpy as np
import dmc_remastered as dmcr

# To maintain reproducibility
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load input args
args = get_dagan_args()
vary = [v for v in dmcr.DMCR_VARY if v != "camera"]
wandb.login()

def create_data(num_of_episodes):
    """
    Returns originals and augmentations from one episode
    """
    originals = []
    augmentations = []

    dynamics_seed = random.randint(1, 1_000_000)
    visual_seed = random.randint(1, 1_000_000)
    env = dmcr.wrapper.make(
        'cheetah',
        'run',
        dynamics_seed=dynamics_seed,
        visual_seed=0,
        vary=vary,
        height=64,
        width=64,
        frame_stack=1
    )
    noisy_env = dmcr.wrapper.make(
        'cheetah',
        'run',
        dynamics_seed=dynamics_seed,
        visual_seed=visual_seed,
        vary=vary,
        height=64,
        width=64,
        frame_stack=1
    )

    for i in range(num_of_episodes):
        env.reset()
        noisy_env.reset()
        done = False

        while not done:
            action = env.action_space.sample()

            # action repeated 20 times
            for _ in range(20):
                next_obs, _, done, _ = env.step(action)
                next_noisy_obs, _, _, _ = noisy_env.step(action)
                originals.append(next_obs)
                augmentations.append(next_noisy_obs)

    return {
        'o': np.array(originals),
        'a': np.array(augmentations),
    }
    
def main():
    args = prepare_args(args)
    
    # init wandb and visualizer
    logger = Logger(args, wandb_project=args.wandb_project_name)

    # if sweep then load sweep config into args
    if args.sweep:
        config = wandb.config
        print(f'Config: {config}')

        # overwrite args with values from sweep config
        for key in config.keys():
            setattr(args, key, config[key])

    model_path, val_path = create_output_dirs(args.model_path, args.name, args.sweep)

    # load the data
    print('Loading the initial..')
    train_data, val_data = load_data(args.dataset.path)
    train_dl = create_dl(train_data['o'], train_data['o'], args.batch_size)
    val_dl = create_dl(val_data['o'], val_data['a'], args.batch_size)

    # init networks and corresponding optimizers
    print('Initialize the networks..')
    g = Generator(
        dim=args.img_size or train_data['o'].shape[2],
        channels=train_data['o'].shape[1],
        dropout_rate=args.dropout_rate,
        device=device
    )
    d = Discriminator(
        dim=args.img_size or train_data['o'].shape[2],
        channels=train_data['o'].shape[1] * 2,
        dropout_rate=args.dropout_rate
    )
    g_opt = optim.Adam(g.parameters(), lr=0.0001, betas=(0.0, 0.9))
    d_opt = optim.Adam(d.parameters(), lr=0.0001, betas=(0.0, 0.9))

    # train the models
    trainer = DaganTrainer(
        generator=g,
        discriminator=d,
        gen_optimizer=g_opt,
        dis_optimizer=d_opt,
        logger=logger,
        device=device,
        critic_iterations=5,
    )

    # initial training 
    print('Start initial training..')
    trainer.train_iteratively(args.initial_epochs, train_dl, val_dl)

    # Save final generator model
    save_model(trainer.g, model_path, prefix='before_tune')

    # Fine tuning iterations
    if args.sweep:
        trainer.gp_weight = args.gp_weight
    print('Start fine-tuning..')
    for i in range(args.max_iterations):
        # create new data
        new_ep_data = create_data(args.trajectories)
        new_train_data = sample_data(train_data, new_ep_data, args.data_ratio)
        
        trainer.train_iteratively(
            args.epochs_per_iteration,
            create_dl(new_train_data['o'], new_train_data['a'], args.batch_size),
            val_dl,
            args.detach,
        )

        # log and update
        trainer.store_augmentations(val_dl, os.path.join(val_path, str(i)))
        update_data(train_data, new_ep_data)

    # final call to visualize the generations of the epochs
    print('Saving validation images on WandB..')
    trainer.logger.log_generation()

    # Save final generator model
    print('Saving models..')
    save_model(trainer.g, model_path)

if __name__ == '__main__':
    if args.sweep:
        sweep_config = sweep_config
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project_name)
        wandb.agent(sweep_id, function=main)
    else:
        main()
