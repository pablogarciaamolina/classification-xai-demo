import streamlit as st
import torch
import numpy as np
from src.core.data import load_dataset_stl10, load_dataset_big_cats

@st.cache_resource
def get_dataset(dataset_name: str) -> torch.utils.data.Dataset:
    """
    Method for caching datasets based on dataset name
    
    Args:
        dataset_name: Name of the dataset ("stl10" or "big_cats")
    
    Returns:
        The loaded dataset
    """
    if dataset_name == "stl10":
        return load_dataset_stl10("test")
    elif dataset_name == "big_cats":
        return load_dataset_big_cats("test")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

@st.cache_resource
def get_dataloader(_dataset: torch.utils.data.Dataset, batch_size: int = 10) -> torch.utils.data.DataLoader:
    """
    Method for caching dataloaders
    
    Args:
        _dataset: The dataset to create a dataloader for
        batch_size: Number of images per batch
    
    Returns:
        The dataloader
    """
    return torch.utils.data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=True
    )

@st.cache_resource
def get_stl10_dataset() -> torch.utils.data.Dataset:
    """
    Method for caching the STL10 dataset (backward compatibility)
    """
    return load_dataset_stl10("test")


@st.cache_resource
def get_stl10_dataloader(_dataset: torch.utils.data.Dataset, batch_size: int = 10) -> torch.utils.data.DataLoader:
    """
    Method for caching the STL10 dataset (backward compatibility)
    """
    return torch.utils.data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=True
    )

def load_new_batch(dataloader: torch.utils.data.DataLoader) -> tuple:
    """
    Method for loading the next batch from a dataloader

    Args:
        dataloader: The dataloader from which to extract the data.
    
    Returns:
        A tuple returned by the returned elements when iterating over the dataloader 
    """

    return next(iter(dataloader))

def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """
    Method for transforming the tensorized data into an streamlit-ready for displaying an image. 

    Args:
        tensor: The tensorized image of shape [1, 3, h, w]

    Returns:
        The transformed numy version.
    """
    return tensor.squeeze().permute(1, 2, 0).numpy()

def get_label_name(dataset: torch.utils.data.Dataset, label_idx: int) -> str:
    """
    Shorcut for getting the correcponding class label given the class index

    Args:
        dataset: The dataset from which to extract the classes.
        label_idx: The label to be mapped into class tag.
    
    Returns:
        The corresponding class tag.
    """

    return dataset.classes[label_idx]
