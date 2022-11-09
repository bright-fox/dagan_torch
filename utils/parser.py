import argparse


def get_dagan_args():
    parser = argparse.ArgumentParser(
        description="Use this script to train a dagan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Filepath for dataset on which to train dagan. File should be .npy format with shape "
        "{"
        "orig: (samples, height, width, channels)"
        "aug: (samples, height, width, channels)"
        "}"
        ,
    )
    parser.add_argument(
        "--model_path", default="./output", type=str, help="Filepath to save final dagan model."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        nargs="?",
        type=int,
        default=32,
        help="batch_size for experiment",
    )
    parser.add_argument(
        "--img_size",
        nargs="?",
        type=int,
        help="Dimension to scale images when training. "
        "Useful when model architecture expects specific input size. "
        "If not specified, uses img_size of data as passed.",
    )
    parser.add_argument(
        "--epochs",
        nargs="?",
        type=int,
        default=50,
        help="Number of epochs to run training.",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        nargs="?",
        default=0.5,
        help="Dropout rate to use within network architecture.",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="dagan",
        help="Name of the model, as well as the wandb run"
    )

    # wandb
    parser.add_argument("--wandb_project_name", default="DAGAN", help="Name for WandB project")
    parser.add_argument('-s', '--sweep', action='store_true', help='do wandb sweep')

    # these arguments are only used for the iterative approach
    parser.add_argument("--initial_epochs", default=20, type=int)
    parser.add_argument("--max_iterations", default=10, type=int)
    parser.add_argument("--epochs_per_iteration", default=1, type=int)
    parser.add_argument("-t", "--trajectories", default=1, type=int, help="number of trajectories to collect per iteration")
    parser.add_argument("--data_ratio", default=1.0, type=float, help="ratio between new and old data with range [0,1]\n1 -> only new 0 -> only old")
    
    parser.add_argument("-d", "--detach", nargs="+", help="networks to freeze", default=[], choices=['gen', 'disc', 'noise'])
    parser.add_argument("-l", "--layer_sizes", nargs="+", help="layers to detach", default=[], type=int)

    return parser.parse_args()

def prepare_args(args):
    """
    Sanity checks and processing for arguments
    """
    if len(args.detach) != len(args.layer_sizes):
        raise ValueError('Detach and amount of layers to detach should correspond to each other')

    # set the layers to detach for networks
    args.detach = {d: args.layer_sizes[i] for i, d in enumerate(args.detach)}

    for network, size in args.detach.items():
        if network == 'gen' and size > 4:
            raise ValueError('Encoder of generator only has 4 layers to freeze')
        if network == 'disc' and size > 4:
            raise ValueError('Discriminator only has 4 layers to freeze')
        if network == 'noise' and size > 3:
            raise ValueError('Noise encoder only has 3 layers to freeze')
    return args
