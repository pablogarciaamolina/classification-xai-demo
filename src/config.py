import torch

# ==================
# USER CONFIGURATION
# ==================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10
batch_size = 64

lr = 1e-4
epochs = 100

# =======================
# DEVELOPER CONFIGURATION
# =======================

NUM_CLASSES = num_classes
BATCH_SIZE = batch_size

# PIPELINE (TRAINING & EVALUATION)

BIG_CATS_PIPELINE_CONFIG = {
    "device": device,
    "epochs": epochs,
    "loss_class": torch.nn.CrossEntropyLoss,
    "optimizer_class": torch.optim.AdamW,
    "optimizer_kwargs": {
        "lr": lr, 
        "weight_decay": 0.01
    },
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_kwargs": {
        "eta_min": 0.00001,
        "T_max": epochs
    }
}

STL10_PIPELINE_CONFIG = {
    "device": device,
    "epochs": epochs,
    "loss_class": torch.nn.CrossEntropyLoss,
    "optimizer_class": torch.optim.AdamW,
    "optimizer_kwargs": {
        "lr": lr, 
        "weight_decay": 0.01
    },
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_kwargs": {
        "eta_min": 0.00001,
        "T_max": epochs
    }
}

PEDIATRIC_PNEUMONIA_PIPELINE_CONFIG = {
    "device": device,
    "epochs": epochs,
    "loss_class": torch.nn.CrossEntropyLoss,
    "optimizer_class": torch.optim.AdamW,
    "optimizer_kwargs": {
        "lr": lr, 
        "weight_decay": 0.01
    },
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_kwargs": {
        "eta_min": 0.00001,
        "T_max": epochs
    }
}