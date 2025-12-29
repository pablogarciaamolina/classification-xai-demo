import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from torch.utils.data import Dataset, DataLoader

from src.core.data._config import PEDIATRIC_PNEUMONIA_TRANSFORMS, PEDIATRIC_PNEUMONIA_TRAINING_TRANSFORMS, DATA_DIR, PEDIATRIC_PNEUMONIA_DIR


class Pediatric_Pneumonia_Dataset(Dataset):
    """
    Dataset for Pediatric Chest X-ray Pneumonia from Kaggle.
    """

    classes = ['NORMAL', 'PNEUMONIA']
    splits = ["train", "test", "mixed"]

    def __init__(self, dir_path, split='train', transform=None):
        """
        Args:
        dir_path: path to the base folder of the dataset (which contains train/ and test/)
        split: 'train', 'test' or 'mixed'
        transform: torchvision transforms to apply to images
        """
        self.transform = transform
        self.images = []

        assert split in self.splits, "Invalid `split` argument"
        for s in ["train", "test"] if split == "mixed" else [split]:
            split_dir = os.path.join(dir_path, s)
            for i in range(len(self.classes)):
                cls_folder = os.path.join(split_dir, self.classes[i])
                self.images += [(os.path.join(cls_folder, fname), i, self._extract_pacient(fname, i)) for fname in os.listdir(cls_folder)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label, _ = self.images[idx]
        img = Image.open(img_path).convert('L')
        
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def _extract_pacient(self, file_name: str, class_idx: int):

        if self.classes[class_idx] == "NORMAL":
            if file_name[:2] == "IM":
                return file_name[:7]
            elif "NORMAL2" in file_name:
                return file_name[:15]
            else:
                return file_name
        elif self.classes[class_idx] == "PNEUMONIA":
            return file_name[:7]
        else:
            raise ValueError
        
def load_dataset_pediatric_pneumonia(split: str = "mixed", for_training: bool = False) -> Pediatric_Pneumonia_Dataset:
    """
    Loads the dataset of Pediatric Pneumonia for a given split of the data

    Args:
        split: What split of the data. Defaults to `mixed`.
        for_training: Whether the dataset is being loaded for training, in order to apply the training transforms (data augmentation). Defaults to `False`.
    
    Returns:
        The instance of the Pedriatric Pneumonia dataset.
    """
    
    return Pediatric_Pneumonia_Dataset(
        os.path.join(DATA_DIR, PEDIATRIC_PNEUMONIA_DIR),
        split=split,
        transform=PEDIATRIC_PNEUMONIA_TRANSFORMS if not for_training else PEDIATRIC_PNEUMONIA_TRAINING_TRANSFORMS
    )

def load_pediatric_pneumonia(
    batch_size: int = 128
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loading method for Peadiatric Chest X-ray Pneumonia dataset.
    Expects the images to be saved inside the directory designed for data (`DATA_DIR`) under the name "Peadiatric Chest X-ray Pneumonia"

    Args:
        batch_size: batch size. Defaults to 128.

    Returns:
        tuple of three dataloaders, train, val and test in respective order.
    """
    
    full_train_dataset = load_dataset_pediatric_pneumonia("train")
    
    test_dataset = load_dataset_pediatric_pneumonia("test")

    # Stratified sampling to ensure each patient goes in a single split and the number of examples per label are balanced
    df = pd.DataFrame(
        full_train_dataset.images,
        columns=["path", "label", "patient"]
    )
    patients = df.groupby("patient")["label"].first().reset_index()
    train_pat, val_pat = train_test_split(
        patients,
        test_size=0.2,
        stratify=patients["label"],
        random_state=42
    )
    train_idx = df[df["patient"].isin(train_pat["patient"])].index.tolist()
    val_idx = df[df["patient"].isin(val_pat["patient"])].index.tolist()
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)

    # train_size = int(len(full_train_dataset) * 0.8)
    # val_size = len(full_train_dataset) - train_size
    # train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return [train_loader, val_loader, test_loader]

def load_pediatric_pneumonia_mixed(
    batch_size: int = 128,
    test_size: float = 0.2,
    val_size: float = 0.1,
    for_training: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Version that mixes original train + test splits and then
    performs stratified group split by patient.

    Args:
        batch_size: batch size. Defaults to 128.
        test_size: size of the test set. Defaults to 0.2.
        val_size: size of the validation set. Defaults to 0.1.
        for_training: whether the dataset is being loaded for training, in order to apply the training transforms (data augmentation). Defaults to `False`.
    """

    full_dataset = load_dataset_pediatric_pneumonia("mixed", for_training=for_training)

    df = pd.DataFrame(
        full_dataset.images,
        columns=["path", "label", "patient"]
    )

    patients = df.groupby("patient")["label"].first().reset_index()

    trainval_pat, test_pat = train_test_split(
        patients,
        test_size=test_size,
        stratify=patients["label"],
        random_state=42
    )

    train_pat, val_pat = train_test_split(
        trainval_pat,
        test_size=val_size / (1 - test_size),
        stratify=trainval_pat["label"],
        random_state=42
    )

    subset_idx = lambda patients_subset: df[df["patient"].isin(patients_subset["patient"])].index.tolist()

    train_idx = subset_idx(train_pat)
    val_idx = subset_idx(val_pat)
    test_idx = subset_idx(test_pat)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader