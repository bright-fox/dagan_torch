import random
from dagan_trainer import DaganTrainer
from dagan_torch.discriminator import Discriminator
from dagan_torch.generator import Generator
from dagan_torch.dataset import create_dl
from utils.parser import get_dagan_args
from utils.visualizer import Visualizer
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

vary = [v for v in dmcr.DMCR_VARY if v != "camera"]

def create_data(batch_size, max_train_size=None):
    """
    Creates data loader for one episode of a random environment
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

    o = np.array(originals)
    a = np.array(augmentations)
    num_val_data = 10

    # shuffle and create train and val dl
    indices = np.arange(o.shape[0])
    np.random.shuffle(indices)

    if max_train_size is None or max_train_size >= o.shape[0] - num_val_data:
        max_train_size = o.shape[0] - num_val_data

    train_dl = create_dl(o[indices[:max_train_size]], a[indices[:max_train_size]], batch_size)
    val_dl = create_dl(o[indices[max_train_size:max_train_size+num_val_data]], a[indices[max_train_size:max_train_size+num_val_data]], batch_size)

    return train_dl, val_dl


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load input args
args = get_dagan_args()

model_path = args.final_model_path
# Input sanity checks
model_dir = os.path.dirname(model_path) or os.getcwd()
if not os.access(model_dir, os.W_OK):
    raise ValueError(model_path + " is not a valid filepath.")

# make dir for validation images
path = os.path.join(model_path, args.name)
os.mkdir(path)

# init wandb
vis = Visualizer(args, wandb_project='DAGAN_iterative_test')

# load the data
initial_train_data = np.load(f'{args.dataset_path}/train.npz')
initial_val_data = np.load(f'{args.dataset_path}/val.npz')
train_dl = create_dl(initial_train_data['orig'], initial_train_data['aug'], args.batch_size)
val_dl = create_dl(initial_val_data['orig'], initial_val_data['aug'], args.batch_size)

# get img info
in_channels = initial_train_data['orig'].shape[1]
img_size = args.img_size or initial_train_data['orig'].shape[2]

# init networks and corresponding optimizers
g = Generator(dim=img_size, channels=in_channels, dropout_rate=args.dropout_rate, device=device)
d = Discriminator(dim=img_size, channels=in_channels * 2, dropout_rate=args.dropout_rate)
g_opt = optim.Adam(g.parameters(), lr=0.0001, betas=(0.0, 0.9))
d_opt = optim.Adam(d.parameters(), lr=0.0001, betas=(0.0, 0.9))

# train the models
trainer = DaganTrainer(
    generator=g,
    discriminator=d,
    gen_optimizer=g_opt,
    dis_optimizer=d_opt,
    visualizer=vis,
    device=device,
    critic_iterations=5,
)

# initial training 
print('Start initial training..')
trainer.train_iteratively(args.initial_epochs, train_dl, val_dl)

# the further iterations
print('Start iterative training..')
for i in range(args.max_iterations):

    train_dl, val_dl = create_data(args.batch_size, args.batch_size * 10)
    trainer.train_iteratively(args.epochs_per_iteration, train_dl, val_dl, detach_encoder=args.detach_encoder)
    trainer.store_augmentations(val_dl, os.path.join(path, str(i)))


# final call to visualize the generations of the epochs
trainer.visualizer.log_generation()

# Save final generator model
torch.save(trainer.g, f'{model_path}/{args.name}_model.pt')
torch.save(trainer.g.state_dict(), f'{model_path}/{args.name}_state_dict.pt')
