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
        "--name",
        type=str,
        default="dagan",
        help="Name of the model, as well as the wandb run"
    )
    parser.add_argument(
        "--use_wandb",
        action='store_true',
        help='use wandb'
    )

    # these arguments are only used for the iterative approach
    parser.add_argument("--epochs_per_iteration", default=1, type=int)
    parser.add_argument("--data_per_iteration", default=1000, type=int, help="Train dataset size per iteration")
    parser.add_argument("--max_iterations", default=10, type=int)
    parser.add_argument("--detach_encoder", action='store_true')
    parser.add_argument("--initial_epochs", default=20)

    return parser.parse_args()
