from torch.utils.data import Dataset, DataLoader

class DaganDataset(Dataset):
    """
    Dataset that contains original images and its corresponding augmented image
    """

    def __init__(self, originals, augmentations):
        assert len(originals) == len(augmentations)
        self.originals = originals
        self.augmentations = augmentations

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):
        """
        Scales original and augmentation to [0, 1] and afterwards normalize it to [-1, 1] and returns
        images as a tuple (original, augmentation)
        """
        # scale to [0, 1], then normalize to [-1, 1]
        return ((self.originals[idx] / 255) - 0.5) / 0.5, ((self.augmentations[idx] / 255) - 0.5) / 0.5

class DaganNormalizedDataset(Dataset):
    """
    Dataset that contains original images and its corresponding augmented image (already normalized to [-1,1])
    """

    def __init__(self, originals, augmentations):
        assert len(originals) == len(augmentations)
        self.originals = originals
        self.augmentations = augmentations

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):
        """
        """
        # scale to [0, 1], then normalize to [-1, 1]
        return self.originals[idx], self.augmentations[idx]

def create_normalized_dl(originals, augmentations, batch_size):
    train_data = DaganNormalizedDataset(originals, augmentations)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)

def create_dl(originals, augmentations, batch_size):
    train_dataset = DaganDataset(originals, augmentations)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)