import os
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import STL10
from src.core.data._config import STL10_TRANSFORMS, DATA_DIR, STL10_DIR, STL10_TRAINING_TRANSFORMS

class STL10_Dataset(STL10):
    classes = [
        "airplane", "bird", "car", "cat", "deer",
        "dog", "horse", "monkey", "ship", "truck"
    ]

def load_dataset_stl10(split: str = "train", download: bool = True, for_training: bool = False):
    """
    Loads the STL10 dataset.
    
    Args:
        split: 'train', 'test', 'unlabeled', 'train+unlabeled'.
        download: If true, downloads the dataset from the internet and puts it in root directory.
        for_training: whether the data is being loaded for training or not, in order to use a different set of transforms. Defaults to False.
    """

    return STL10_Dataset(
        root=os.path.join(DATA_DIR, STL10_DIR),
        split=split,
        download=download,
        transform=STL10_TRAINING_TRANSFORMS if for_training and split == "train" else STL10_TRANSFORMS
    )

def load_stl10(
    batch_size: int = 128,
    val_split: float = 0.1,
    for_training: bool = False
) -> list[DataLoader]:
    """
    Loading method for STL10 dataset.
    
    Args:
        batch_size: batch size. Defaults to 128.
        val_split: fraction of training set to use for validation. Defaults to 0.1.
        for_training: whether the data is being loaded for training or not. Defaults to False.

    Returns:
        list of three dataloaders: train, val, test.
    """
    
    full_train_dataset = load_dataset_stl10(split="train", download=True, for_training=for_training)
    test_dataset = load_dataset_stl10(split="test", download=True, for_training=False)
    
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return [train_loader, val_loader, test_loader]
