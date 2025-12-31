import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from src.core.data._config import GARBAGE_TRANSFORMS, DATA_DIR, GARBAGE_DIR, GARBAGE_TRAINING_TRANSFORMS


class Garbage_Dataset(Dataset):
    """
    Dataset for Garbage Dataset from Kaggle.
    """

    classes = [
        "battery",
        "biological",
        "cardboard",
        "clothes",
        "glass",
        "metal",
        "paper",
        "plastic",
        "shoes",
        "trash"
    ]

    def __init__(self, dir_path, transforms=None):
        """
        Args:
            dir_path: path to the base folder of the dataset
            transforms: torchvision transforms to apply to images
        """

        self.transforms = transforms
        self.images = []

        for i in range(len(self.classes)):
            cls_folder = os.path.join(dir_path, self.classes[i])
            self.images += [(os.path.join(cls_folder, fname), i, fname[:-4]) for fname in os.listdir(cls_folder)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label, _ = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transforms:
            img = self.transforms(img)
        return img, label
        
def load_dataset_garbage(for_training: bool = False) -> Garbage_Dataset:
    """
    Loads the dataset of Garbage for a given split of the data

    Args:
        split: Split of the data. Defaults to `train`.
        for_training: whether the data is being loaded for training or not, in order to use a different set of transforms. Defaults to False.
    
    Returns:
        The instance of the Garbage dataset.
    """
    
    return Garbage_Dataset(
        os.path.join(DATA_DIR, GARBAGE_DIR),
        transforms= GARBAGE_TRAINING_TRANSFORMS if for_training else GARBAGE_TRANSFORMS
    )

def load_garbage(
    batch_size: int = 128,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    for_training: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loading method for Garbage dataset.
    Expects the images to be saved inside the directory designed for data (`DATA_DIR`) under the name `GARBAGE_DIR`

    Args:
        batch_size: batch size. Defaults to 128.
        train_size: size of the training set. Defaults to 0.7.
        val_size: size of the validation set. Defaults to 0.1.
        test_size: size of the test set. Defaults to 0.2.
        for_training: whether the data is being loaded for training or not. Defaults to False.

    Returns:
        tuple of three dataloaders, train, val and test in respective order.
    """
    
    full_dataset = load_dataset_garbage(for_training)
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return [train_loader, val_loader, test_loader]