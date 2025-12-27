import streamlit as st
import torch
import numpy as np
from src.core.data import load_dataset_big_cats

@st.cache_resource
def get_big_cats_dataset() -> torch.utils.data.Dataset:
    """
    Method for caching the Big Cats dataset
    """
    return load_dataset_big_cats("test")


@st.cache_resource
def get_big_cats_dataloader(_dataset: torch.utils.data.Dataset, batch_size: int = 10) -> torch.utils.data.DataLoader:
    """
    Method for caching the Big Cats dataset
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
