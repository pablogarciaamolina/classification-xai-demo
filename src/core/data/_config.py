from torchvision import transforms
from ._utils import RandomPatch

DATA_DIR: str = "data"

BIG_CATS_DIR = "10 Big Cats"
BIG_CATS_IMAGE_SIZE = (200, 200)
BIG_CATS_TRANSFORMS = transforms.Compose([
    transforms.Resize(BIG_CATS_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
BIG_CATS_TRAINING_TRANSFORMS = transforms.Compose([
    transforms.RandomCrop(BIG_CATS_IMAGE_SIZE),
    RandomPatch(patch_size=(40, 40), probability=0.5, color=(0, 0, 0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomInvert(0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])