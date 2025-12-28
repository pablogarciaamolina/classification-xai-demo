import torch
from captum.attr import IntegratedGradients
from typing import Optional


class Integrated_Gradients:
    """
    Integrated Gradients implementation using Captum library.
    """
    
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Initialize Integrated Gradients.
        
        Args:
            model: PyTorch model to explain
        """
        self.model = model
        self.ig = IntegratedGradients(model)
 
    def explain(
        self, 
        input_tensor: torch.Tensor, 
        class_idx: int,
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate Integrated Gradients explanation for input image.
        
        Args:
            input_tensor: Input image tensor of shape (1, C, H, W)
            class_idx: Target class index
            n_steps: Number of steps in the Riemann approximation
            baseline: Baseline image (if None, uses zero baseline)
        
        Returns:
            Attribution map of shape (H, W), normalized to [0, 1]
        """
        
        attributions = self.ig.attribute(
            input_tensor, 
            target=class_idx, 
            n_steps=n_steps,
            baselines=baseline
        )
        
        g = attributions.sum(dim=1).squeeze().cpu().detach()
        g = g - g.min()
        g /= (g.max() + 1e-8)
        
        return g