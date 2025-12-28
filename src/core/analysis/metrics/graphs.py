import os
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import torch

from src.core.analysis._config import ANALYSIS_DIR, GRAPHS_DIR

def confusion_matrix(
    y_pred: torch.Tensor,
    y_test: torch.Tensor,
    save_name: str = 'confusion_matrix',
    labels: Optional[list[str]] = None
):
    """
    Plots the confusion matrix for the given predicted and true values.

    Args:
        y_pred: Predicted values. Tensor of shape [batch]
        y_test: True values. Tensor of shape [batch]
        save_name: Name of the file to save the confusion matrix.
        labels: Optional list of labels to use for mapping the labels in the confusion matrix. If not provided, the labels will be inferred from the data.
    
    Returns:
        The confusion matrix figure object.
    """

    assert y_pred.shape == y_test.shape
    assert y_pred.ndim == 1

    y_pred = y_pred.cpu().numpy()
    y_test = y_test.cpu().numpy()

    cm = metrics.confusion_matrix(y_test, y_pred)

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels if labels is not None else 'auto',
                yticklabels=labels if labels is not None else 'auto')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    save_name = save_name.replace('/', '_').replace('\\', '_')    
    if not save_name.endswith(".png"):
        save_name += ".png"
    
    os.makedirs(os.path.join(ANALYSIS_DIR, GRAPH_DIR), exist_ok=True)
    plt.savefig(os.path.join(ANALYSIS_DIR, GRAPH_DIR, save_name))
    
    return fig