from dagan_trainer import DaganTrainer
from dagan_torch.discriminator import Discriminator
from dagan_torch.generator import Generator
from dagan_torch.dataset import create_dl
from utils.parser import get_dagan_args, prepare_args
import torch
import os
import torch.optim as optim
import numpy as np
from utils.logger import Logger
from utils.utils import create_output_dirs

# To maintain reproducibility
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load input args
args = get_dagan_args()
args = prepare_args(args)

model_path, val_path = create_output_dirs(args.model_path, args.name, is_sweep=False)

# init wandb
logger = Logger(args)

# load the data
train_data = np.load(os.path.join(args.dataset_path, 'train.npz'))
val_data = np.load(os.path.join(args.dataset_path, 'val.npz'))
train_dl = create_dl(train_data['orig'], train_data['aug'], args.batch_size)
val_dl = create_dl(val_data['orig'], val_data['aug'], args.batch_size)

# get img info
in_channels = train_data['orig'].shape[1]
img_size = args.img_size or train_data['orig'].shape[2]

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
    logger=logger,
    device=device,
    critic_iterations=5,
)
trainer.train(
    data_loader=train_dl,
    epochs=args.epochs,
    val_dataloader=val_dl,
)

# Save final generator model
torch.save(trainer.g, os.path.join(model_path, 'model.pt'))
torch.save(trainer.g.state_dict(), os.path.join(model_path, 'state_dict.pt'))

# Save the discriminator model
torch.save(trainer.d, os.path.join(model_path, 'disc.pt'))
torch.save(trainer.d.state_dict(), os.path.join(model_path, 'disc_state_dict.pt'))