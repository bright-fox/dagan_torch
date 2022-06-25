from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings


class DaganDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, originals, augmentations, transform=None):
        assert len(originals) == len(augmentations)
        self.originals = originals
        self.augmentations = augmentations
        self.transform = transform

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.transform(self.originals[idx]), self.transform(
                self.augmentations[idx]
            )


def create_dagan_dataloader(originals, augmentations, transform, batch_size):
    train_dataset = DaganDataset(originals, augmentations, transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
