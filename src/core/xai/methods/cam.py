import torch

class GradCAM:
    """
    Grad-CAM implementation.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer_name: str) -> None:
        
        self.model = model

        self.cnn_outputs = None
        self.last_cnn_grad = None

        target_module = dict(self.model.named_modules())[target_layer_name]
        target_module.register_forward_hook(self._forward_hook)
        target_module.register_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _input, output):
        self.cnn_outputs = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.last_cnn_grad = grad_output[0].detach()

    def explain(self, inputs: torch.Tensor, class_idx: int = None) -> torch.Tensor:
        """
        Method for explaining a single input image.

        Args:
            inputs: Input image. Must have the shape expected by the model, batched. [1, channels, H, W]
            class_idx (int, optional): Class index to explain. Defaults to None, in which case the class with the highest probability is used.

        Returns:
            torch.Tensor: Resized explanation of shape [H, W]
        """

        assert inputs.dim() == 4

        probabilities = self.model(inputs)  # [batch, num_classes]
        if class_idx is None:
            class_probs, _ = torch.max(probabilities, dim=-1)  # [batch]
        else:
            assert type(class_idx) == int
            class_probs = probabilities[:, class_idx]  # [batch]

        self.model.zero_grad()
        class_probs.sum().backward(retain_graph=True)

        weights = torch.nn.functional.adaptive_avg_pool2d(
            self.last_cnn_grad, (1, 1)
        )  # [batch, out_channels]

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
