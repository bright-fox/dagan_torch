from dagan_trainer import DaganTrainer
from discriminator import Discriminator
from generator import Generator
from dataset import create_dagan_dataloader
from utils.parser import get_dagan_args
import torchvision.transforms as transforms
import torch
import os
import torch.optim as optim
import numpy as np


# To maintain reproducibility
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load input args
args = get_dagan_args()

# get the data set paths
dataset_path = args.dataset_path
train_dataset_path = f'{dataset_path}/train.npz'
test_dataset_path = f'{dataset_path}/test.npz'
val_dataset_path = f'{dataset_path}/val.npz'

# load the data
train_data = np.load(train_dataset_path)
val_data = np.load(val_dataset_path)

final_generator_path = args.final_model_path
save_checkpoint_path = args.save_checkpoint_path
load_checkpoint_path = args.load_checkpoint_path
in_channels = train_data['orig'].shape[-1]
img_size = args.img_size or train_data['orig'].shape[2]
batch_size = args.batch_size
epochs = args.epochs
dropout_rate = args.dropout_rate
max_pixel_value = args.max_pixel_value
should_display_generations = not args.suppress_generations

# Input sanity checks
final_generator_dir = os.path.dirname(final_generator_path) or os.getcwd()
if not os.access(final_generator_dir, os.W_OK):
    raise ValueError(final_generator_path + " is not a valid filepath.")

g = Generator(dim=img_size, channels=in_channels, dropout_rate=dropout_rate)
d = Discriminator(dim=img_size, channels=in_channels * 2, dropout_rate=dropout_rate)

mid_pixel_value = max_pixel_value / 2
train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (mid_pixel_value,) * in_channels, (mid_pixel_value,) * in_channels # mean, standard deviation
        ),
    ]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataloader = create_dagan_dataloader(train_data['orig'], train_data['aug'], train_transform, batch_size)

g_opt = optim.Adam(g.parameters(), lr=0.0001, betas=(0.0, 0.9))
d_opt = optim.Adam(d.parameters(), lr=0.0001, betas=(0.0, 0.9))

display_transform = train_transform

trainer = DaganTrainer(
    generator=g,
    discriminator=d,
    gen_optimizer=g_opt,
    dis_optimizer=d_opt,
    batch_size=batch_size,
    device=device,
    critic_iterations=5,
    print_every=75,
    num_tracking_images=0,
    save_checkpoint_path=save_checkpoint_path,
    load_checkpoint_path=load_checkpoint_path,
    display_transform=display_transform,
    should_display_generations=should_display_generations,
)
trainer.train(data_loader=train_dataloader, epochs=epochs, val_images=val_data['orig'])

# Save final generator model
torch.save(trainer.g, final_generator_path)
