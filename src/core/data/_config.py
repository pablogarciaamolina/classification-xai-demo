from torchvision import transforms
from ._utils import RandomPatch

DATA_DIR = "data"

# BIG CATS
BIG_CATS_DIR = "10 Big Cats"
BIG_CATS_IMAGE_SIZE = (200, 200)
BIG_CATS_TRANSFORMS = transforms.Compose([
    transforms.Resize(BIG_CATS_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
BIG_CATS_TRAINING_TRANSFORMS = transforms.Compose([
    transforms.RandomCrop(BIG_CATS_IMAGE_SIZE),
    # RandomPatch(patch_size=(40, 40), probability=0.5, color=(0, 0, 0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    # transforms.RandomInvert(0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# GARBAGE
GARBAGE_DIR = "Garbage Dataset"
GARBAGE_IMAGE_SIZE = (200, 200)
GARBAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize(GARBAGE_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
GARBAGE_TRAINING_TRANSFORMS = transforms.Compose([
    transforms.Resize(GARBAGE_IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomInvert(0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# PEDIATRIC PNEUMONIA
PEDIATRIC_PNEUMONIA_DIR = "Pediatric Chest X-ray Pneumonia"
PEDIATRIC_PNEUMONIA_IMAGE_SIZE = (220, 220)
PEDIATRIC_PNEUMONIA_TRANSFORMS = transforms.Compose([
    transforms.Resize(PEDIATRIC_PNEUMONIA_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
PEDIATRIC_PNEUMONIA_TRAINING_TRANSFORMS = transforms.Compose([
    transforms.CenterCrop(PEDIATRIC_PNEUMONIA_IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# STL10
STL10_DIR = "stl10"
STL10_IMAGE_SIZE = (96, 96)
STL10_TRANSFORMS = transforms.Compose([
    transforms.Resize(STL10_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
STL10_TRAINING_TRANSFORMS = transforms.Compose([
    transforms.RandomCrop(STL10_IMAGE_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
