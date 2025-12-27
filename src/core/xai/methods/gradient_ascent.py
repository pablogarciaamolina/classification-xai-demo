import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GradientAscent:
    """
    Gradient Ascent for visualizing what a neural network has learned for a specific class.
    
    Optimizes input image to maximize activation of a target class logit.
    """
    
    def __init__(self, model, target_class, img_size=(200, 200)):
        """
        Args:
            model: PyTorch model (should be in eval mode)
            target_class: Index of the target class to visualize
            img_size: Tuple (height, width) of the image
        """
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.img_size = img_size
        self.device = next(model.parameters()).device
        
        # Normalization parameters (should match training preprocessing)
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        
    def normalize(self, img):
        """
        Normalize image using training statistics.
        
        Args:
            img: Tensor of shape (1, C, H, W) in range [0, 1]
        
        Returns:
            Normalized image
        """
        return (img - self.mean) / self.std
        
    def total_variation_loss(self, img):
        """
        Compute Total Variation loss for smoothness regularization.
        
        Args:
            img: Tensor of shape (1, C, H, W)
        
        Returns:
            TV loss value
        """
        h_var = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        w_var = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return h_var + w_var
    
    def l2_norm(self, img):
        """
        Compute L2 norm of the image.
        
        Args:
            img: Tensor of shape (1, C, H, W)
        
        Returns:
            L2 norm value
        """
        return torch.norm(img)
    
    def gaussian_blur(self, img, kernel_size=3, sigma=0.5):
        """
        Apply Gaussian blur to the image.
        
        Args:
            img: Tensor of shape (1, C, H, W)
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation of the Gaussian
        
        Returns:
            Blurred image
        """
        channels = img.shape[1]
        
        # Create Gaussian kernel
        kernel = self._get_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.repeat(channels, 1, 1, 1).to(img.device)
        
        # Apply depthwise convolution
        padding = kernel_size // 2
        blurred = F.conv2d(img, kernel, padding=padding, groups=channels)
        
        return blurred
    
    def _get_gaussian_kernel(self, kernel_size, sigma):
        """Create a 2D Gaussian kernel."""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def initialize_image(self, init_type='random'):
        """
        Initialize the input image.
        
        Args:
            init_type: 'random' or 'gray'
        
        Returns:
            Initialized image tensor of shape (1, 3, H, W)
        """
        if init_type == 'random':
            # Random initialization with small values around 0.5
            img = torch.randn(1, 3, *self.img_size) * 0.1 + 0.5
        elif init_type == 'gray':
            # Gray image
            img = torch.ones(1, 3, *self.img_size) * 0.5
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        
        img = img.to(self.device)
        img.requires_grad = True
        return img
    
    def generate(self, 
        iterations=300,
        lr=0.1,
        l2_weight=0.001,
        tv_weight=0.01,
        blur_freq=10,
        blur_sigma=0.5,
        init_type='random',
        jitter=0,
        verbose=True
    ):
        """
        Generate an image that maximizes the target class activation.
        
        Args:
            iterations: Number of optimization steps
            lr: Learning rate
            l2_weight: Weight for L2 regularization
            tv_weight: Weight for Total Variation regularization
            blur_freq: Apply blur every N iterations (0 to disable)
            blur_sigma: Sigma for Gaussian blur
            init_type: 'random' or 'gray'
            jitter: Random jitter range in pixels (for translation invariance)
            verbose: Print progress
        
        Returns:
            Optimized image as numpy array (H, W, C) in [0, 255]
        """
        img = self.initialize_image(init_type)
        optimizer = torch.optim.Adam([img], lr=lr)
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            if jitter > 0:
                ox = np.random.randint(-jitter, jitter + 1)
                oy = np.random.randint(-jitter, jitter + 1)
                img_jittered = torch.roll(img, shifts=(ox, oy), dims=(2, 3))
            else:
                img_jittered = img
            
            img_normalized = self.normalize(img_jittered)
            
            output = self.model(img_normalized)
            
            loss = -output[0, self.target_class]
            
            if l2_weight > 0:
                loss += l2_weight * self.l2_norm(img)
            
            if tv_weight > 0:
                loss += tv_weight * self.total_variation_loss(img)
            
            loss.backward()
            optimizer.step()
            
           with torch.no_grad():
                img.clamp_(0, 1)
            
            if blur_freq > 0 and (i + 1) % blur_freq == 0:
                with torch.no_grad():
                    img.data = self.gaussian_blur(img, sigma=blur_sigma)
            
            if verbose and (i + 1) % 50 == 0:
                class_score = output[0, self.target_class].item()
                print(f"Iteration {i+1}/{iterations}, Class score: {class_score:.4f}")
        
        return img
    
    def visualize(self, img, title=None, save_path=None):
        """
        Visualize the generated image.
        
        Args:
            img: PyTorch tensor of shape (1, C, H, W)
            title: Title for the plot
            save_path: Path to save the image (optional)
        """

        img_np = img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.axis('off')
        if title:
            plt.title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()