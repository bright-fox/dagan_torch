from dagan_trainer import DaganTrainer
from discriminator import Discriminator
from generator import Generator
from dataset import create_dagan_dataloader
from utils.parser import get_dagan_args
import torch
import os
import torch.optim as optim
import numpy as np
from utils.visualizer import Visualizer

# To maintain reproducibility
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load input args
args = get_dagan_args()
final_generator_path = args.final_model_path

# Input sanity checks
final_generator_dir = os.path.dirname(final_generator_path) or os.getcwd()
if not os.access(final_generator_dir, os.W_OK):
    raise ValueError(final_generator_path + " is not a valid filepath.")

# init wandb
visualizer = Visualizer(args)

# load the data
train_data = np.load(f'{args.dataset_path}/train.npz')
val_data = np.load(f'{args.dataset_path}/val.npz')
train_dataloader = create_dagan_dataloader(train_data['orig'], train_data['aug'], args.batch_size)
val_dataloader = create_dagan_dataloader(val_data['orig'], val_data['aug'], args.batch_size)

# get img info
in_channels = train_data['orig'].shape[1]
img_size = args.img_size or train_data['orig'].shape[2]

# init networks and corresponding optimizers
g = Generator(dim=img_size, channels=in_channels, dropout_rate=args.dropout_rate)
d = Discriminator(dim=img_size, channels=in_channels * 2, dropout_rate=args.dropout_rate)
g_opt = optim.Adam(g.parameters(), lr=0.0001, betas=(0.0, 0.9))
d_opt = optim.Adam(d.parameters(), lr=0.0001, betas=(0.0, 0.9))

# train the models
trainer = DaganTrainer(
    generator=g,
    discriminator=d,
    gen_optimizer=g_opt,
    dis_optimizer=d_opt,
    visualizer=visualizer,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    critic_iterations=5,
)
trainer.train(
    data_loader=train_dataloader,
    epochs=args.epochs,
    val_dataloader=val_dataloader,
)

# Save final generator model
torch.save(trainer.g, f'{final_generator_path}/{args.name}_model.pt')
torch.save(trainer.g.state_dict(), f'{final_generator_path}/{args.name}_state_dict.pt')
