import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple, Literal, Optional

from src.core.analysis._config import ANALYSIS_DIR, XAI_DIR


class Gradient_Ascent:
    """
    Gradient Ascent for visualizing what a neural network has learned for a specific class.
    """
    
    def __init__(self, model: torch.nn.Module, target_class: int, img_size: Tuple[int, int]) -> None:
        """
        Initialize Gradient Ascent visualizer.
        
        Args:
            model: PyTorch model (should be in eval mode)
            target_class: Index of the target class to visualize
            img_size: Tuple (height, width) of the generated image
        """

        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.img_size = img_size
        self.device = next(model.parameters()).device
        
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        
    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalize image using training statistics.
        
        Args:
            img: Tensor of shape [1, C, H, W] in range [0, 1]
        
        Returns:
            Normalized image tensor
        """

        return (img - self.mean) / self.std
    
    def _l2_norm(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 norm of the image.
        
        Args:
            img: Tensor of shape (1, C, H, W)
        
        Returns:
            L2 norm value as tensor
        """

        return torch.norm(img)
    
    def _total_variation_loss(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss for image smoothness.
        
        Args:
            img: Tensor of shape (1, C, H, W)
        
        Returns:
            Total variation loss value
        """

        diff_i = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        diff_j = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        return torch.sum(diff_i) + torch.sum(diff_j)
    
    def _gaussian_blur(self, img: torch.Tensor, kernel_size: int = 3, sigma: float = 0.5) -> torch.Tensor:
        """
        Apply Gaussian blur to the image.
        
        Args:
            img: Tensor of shape (1, C, H, W)
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation of the Gaussian
        
        Returns:
            Blurred image tensor
        """

        channels = img.shape[1]
        
        kernel = self._get_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.repeat(channels, 1, 1, 1).to(img.device)
        
        padding = kernel_size // 2
        blurred = F.conv2d(img, kernel, padding=padding, groups=channels)
        
        return blurred
    
    def _get_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        Create a 2D Gaussian kernel.
        
        Args:
            kernel_size: Size of the kernel
            sigma: Standard deviation
        
        Returns:
            Gaussian kernel tensor of shape (1, 1, kernel_size, kernel_size)
        """

        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _initialize_image(
        self, 
        init_type: Literal['random', 'gray'] = 'random'
    ) -> torch.Tensor:
        """
        Initialize the input image.
        
        Args:
            init_type: 'random' for random noise or 'gray' for uniform gray
        
        Returns:
            Initialized image tensor of shape (1, 3, H, W)
        """

        if init_type == 'random':
            img = torch.randn(1, 3, *self.img_size) * 0.1 + 0.5
        elif init_type == 'gray':
            img = torch.ones(1, 3, *self.img_size) * 0.5
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        
        img = img.to(self.device)
        img.requires_grad = True
        return img
    
    def generate(
        self, 
        iterations: int = 300,
        lr: float = 0.1,
        l2_weight: float = 0.001,
        tv_weight: float = 0.01,
        blur_freq: int = 10,
        blur_sigma: float = 0.5,
        init_type: Literal['random', 'gray'] = 'random',
        jitter: int = 0,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Generate an image that maximizes the target class activation.
        
        Args:
            iterations: Number of optimization steps
            lr: Learning rate
            l2_weight: Weight for L2 regularization
            tv_weight: Weight for Total Variation regularization
            blur_freq: Apply blur every N iterations (0 to disable)
            blur_sigma: Sigma for Gaussian blur
            init_type: 'random' or 'gray' initialization
            jitter: Random jitter range in pixels (for translation invariance)
            verbose: Whether to print progress
        
        Returns:
            Optimized image tensor of shape (1, 3, H, W)
        """

        img = self._initialize_image(init_type)
        optimizer = torch.optim.Adam([img], lr=lr)
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            if jitter > 0:
                ox = torch.randint(-jitter, jitter + 1, (1,)).item()
                oy = torch.randint(-jitter, jitter + 1, (1,)).item()
                img_jittered = torch.roll(img, shifts=(ox, oy), dims=(2, 3))
            else:
                img_jittered = img
            
            img_normalized = self._normalize(img_jittered)
            
            output = self.model(img_normalized)
            
            loss = -output[0, self.target_class]
            
            if l2_weight > 0:
                loss += l2_weight * self._l2_norm(img)
            
            if tv_weight > 0:
                loss += tv_weight * self._total_variation_loss(img)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                img.clamp_(0, 1)
            
            if blur_freq > 0 and (i + 1) % blur_freq == 0:
                with torch.no_grad():
                    img.data = self._gaussian_blur(img, sigma=blur_sigma)
            
            if verbose and (i + 1) % 50 == 0:
                class_score = output[0, self.target_class].item()
                print(f"Iteration {i+1}/{iterations}, Class score: {class_score:.4f}")
        
        return img
    
    def visualize(
        self, 
        img: torch.Tensor, 
        title: Optional[str] = None, 
        save_path: Optional[str] = None, 
        show: bool = False
    ) -> None:
        """
        Visualize the generated image.
        
        Args:
            img: PyTorch tensor of shape (1, C, H, W)
            title: Title for the plot
            save_path: Path to save the image (optional)
            show: Whether to display the plot
        
        Returns:
            None
        """
        
        img_display = img.detach().cpu().squeeze().permute(1, 2, 0)
        img_display = torch.clamp(img_display * 255, 0, 255)
        img_np = img_display.numpy().astype('uint8')

        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.axis('off')
        if title:
            plt.title(title)
        
        if save_path:
            os.makedirs(os.path.join(ANALYSIS_DIR, XAI_DIR), exist_ok=True)
            plt.savefig(os.path.join(ANALYSIS_DIR, XAI_DIR, save_path), bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()