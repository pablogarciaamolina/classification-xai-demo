import torch
from typing import Optional


class Grad_CAM:

    def __init__(self, model: torch.nn.Module, target_layer_name: str) -> None:
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model to explain
            target_layer_name: Name of the target convolutional layer
        """

        self.model = model
        self.cnn_outputs: Optional[torch.Tensor] = None
        self.last_cnn_grad: Optional[torch.Tensor] = None

        target_module = dict(self.model.named_modules())[target_layer_name]
        target_module.register_forward_hook(self._forward_hook)
        target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module: torch.nn.Module, _input: torch.Tensor, output: torch.Tensor) -> None:
        """
        Forward hook to capture layer outputs.
        """
        
        self.cnn_outputs = output.detach()

    def _backward_hook(self, _module: torch.nn.Module, _grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
        """
        Backward hook to capture gradients.
        """
        
        self.last_cnn_grad = grad_output[0].detach()

    def explain(self, inputs: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        """
        Generate Grad-CAM explanation for input image.
        
        Args:
            inputs: Input image tensor of shape [1, C, H, W]
            class_idx: Target class index. If None, uses the predicted class
        
        Returns:
            Grad-CAM heatmap of shape [H, W], normalized to [0, 1]
        """

        assert inputs.dim() == 4, f"Expected 4D input tensor, got {inputs.dim()}D"

        probabilities = self.model(inputs)
        
        if class_idx is None:
            class_probs, _ = torch.max(probabilities, dim=-1)
        else:
            assert isinstance(class_idx, int), f"class_idx must be int, got {type(class_idx)}"
            class_probs = probabilities[:, class_idx]

        self.model.zero_grad()
        class_probs.sum().backward(retain_graph=True)

        weights = torch.nn.functional.adaptive_avg_pool2d(
            self.last_cnn_grad, (1, 1)
        )

        result = torch.nn.functional.relu(torch.sum(weights * self.cnn_outputs, dim=1))

        g = result.detach().cpu()
        g = torch.nn.functional.interpolate(
            g.unsqueeze(1),
            size=inputs.shape[2:],
            mode="bilinear",
            align_corners=False
        ).squeeze()
    
        g = g - g.min()
        g /= (g.max() + 1e-8)

        return g
