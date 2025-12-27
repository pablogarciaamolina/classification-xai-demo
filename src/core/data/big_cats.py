import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from src.core.data._config import BIG_CATS_TRANSFORMS, DATA_DIR, BIG_CATS_DIR, BIG_CATS_TRAINING_TRANSFORMS


class Big_Cats_Dataset(Dataset):
    """
    Dataset for 10 Big Cats from Kaggle.
    """

    classes = [
        "AFRICAN LEOPARD",
        "CARACAL",
        "CHEETAH",
        "CLOUDED LEOPARD",
        "JAGUAR",
        "LIONS",
        "OCELOT",
        "PUMA",
        "SNOW LEOPARD",
        "TIGER",
    ]
    splits = ["train", "test", "valid"]

    def __init__(self, dir_path, split='train', transforms=None):
        """
        Args:
            dir_path: path to the base folder of the dataset
            split: 'train', 'test' or 'valid'
            transforms: torchvision transforms to apply to images
        """

        self.transforms = transforms
        self.images = []

        assert split in self.splits, "Invalid `split` argument"
        for s in [split]:
            split_dir = os.path.join(dir_path, s)
            for i in range(len(self.classes)):
                cls_folder = os.path.join(split_dir, self.classes[i])
                self.images += [(os.path.join(cls_folder, fname), i, fname[:-3]) for fname in os.listdir(cls_folder)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label, _ = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transforms:
            img = self.transforms(img)
        return img, label
        
def load_dataset_big_cats(split: str = "train", for_training: bool = False) -> Big_Cats_Dataset:
    """
    Loads the dataset of Big Cats for a given split of the data

    Args:
        split: Split of the data. Defaults to `train`.
        for_training: whether the data is being loaded for training or not, in order to use a different set of transforms. Defaults to False.
    
    Returns:
        The instance of the Big Cats dataset.
    """
    
    return Big_Cats_Dataset(
        os.path.join(DATA_DIR, BIG_CATS_DIR),
        split=split,
        transforms= BIG_CATS_TRAINING_TRANSFORMS if for_training else BIG_CATS_TRANSFORMS
    )

def load_big_cats(
    batch_size: int = 128,
    for_training: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loading method for Big Cats dataset.
    Expects the images to be saved inside the directory designed for data (`DATA_DIR`) under the name `BIG_CATS_DIR`

    Args:
        batch_size: batch size. Defaults to 128.
        for_training: whether the data is being loaded for training or not. Defaults to False.

    Returns:
        tuple of three dataloaders, train, val and test in respective order.
    """
    
    train_dataset = load_dataset_big_cats("train", for_training)
    val_dataset = load_dataset_big_cats("valid", for_training)
    test_dataset = load_dataset_big_cats("test", for_training)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return [train_loader, val_loader, test_loader]