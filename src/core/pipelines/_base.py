from dataclasses import dataclass, field
import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Type, Optional, Dict, Any

@dataclass
class Pipeline_Config:
    """
    Dataclass for the configuration of a classification pipeline.
    """
    
    # ----- TRAINING HYPERPARAMETERS -----
    epochs: int

    # ----- DEVICE -----
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ----- LOSS CONFIG -----
    loss_class: Type[_Loss] = torch.nn.CrossEntropyLoss

    # ----- OPTIMIZER CONFIG -----
    optimizer_class: Type[Optimizer] = torch.optim.AdamW
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-3,
            "weight_decay": 0.01
        }
    )

    # ----- SCHEDULER CONFIG -----
    scheduler_class: Optional[Type[_LRScheduler]] = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "T_max": 100,
            "eta_min": 1e-5,
        }
    )
