import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, List, Tuple, Dict, Union, Literal
from tqdm import tqdm


class AverageSensitivity:
    """
    Average Sensitivity - Measures the robustness of explanation methods.
    
    Evaluates how much an explanation changes when the input is perturbed slightly.
    A robust explanation method should produce similar explanations for similar inputs.
    
    Reference: Yeh et al. "On the (In)fidelity and Sensitivity of Explanations" (NeurIPS 2019)
    """
    
    def __init__(
        self, 
        explanation_method: Callable[[torch.Tensor, int], Union[np.ndarray, torch.Tensor]], 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """
        Initialize Average Sensitivity evaluator.
        
        Args:
            explanation_method: Function that takes (image, target_class) and returns saliency map
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.explanation_method = explanation_method
        self.device = device
    
    def _add_noise(
        self, 
        image: torch.Tensor, 
        noise_std: float = 0.1, 
        noise_type: Literal['gaussian', 'uniform'] = 'gaussian'
    ) -> torch.Tensor:
        """
        Add noise to the input image.
        
        Args:
            image: Input image tensor of shape (1, C, H, W)
            noise_std: Standard deviation of the noise
            noise_type: Type of noise - 'gaussian' or 'uniform'
        
        Returns:
            Noisy image tensor with same shape as input (1, C, H, W)
        """
        if noise_type == 'gaussian':
            noise = torch.randn_like(image) * noise_std
        elif noise_type == 'uniform':
            noise = (torch.rand_like(image) - 0.5) * 2 * noise_std
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0, 1)
        
        return noisy_image
    
    def _compute_explanation_distance(
        self, 
        exp1: Union[np.ndarray, torch.Tensor], 
        exp2: Union[np.ndarray, torch.Tensor], 
        distance_metric: Literal['l1', 'l2', 'correlation', 'cosine'] = 'l1'
    ) -> float:
        """
        Compute distance between two explanations.
        
        Args:
            exp1: First explanation, shape (H, W) or (1, 1, H, W)
            exp2: Second explanation, shape (H, W) or (1, 1, H, W)
            distance_metric: Distance metric - 'l1', 'l2', 'correlation', or 'cosine'
        
        Returns:
            Distance value as float
        """
        # Convert to tensors if numpy arrays
        if isinstance(exp1, np.ndarray):
            exp1 = torch.from_numpy(exp1).float()
        if isinstance(exp2, np.ndarray):
            exp2 = torch.from_numpy(exp2).float()
        
        # Flatten explanations
        exp1_flat = exp1.view(-1)
        exp2_flat = exp2.view(-1)
        
        if distance_metric == 'l1':
            return torch.mean(torch.abs(exp1_flat - exp2_flat)).item()
        
        elif distance_metric == 'l2':
            return torch.sqrt(torch.mean((exp1_flat - exp2_flat) ** 2)).item()
        
        elif distance_metric == 'correlation':
            exp1_centered = exp1_flat - exp1_flat.mean()
            exp2_centered = exp2_flat - exp2_flat.mean()
            
            correlation = torch.sum(exp1_centered * exp2_centered) / (
                torch.sqrt(torch.sum(exp1_centered ** 2)) * 
                torch.sqrt(torch.sum(exp2_centered ** 2)) + 1e-8
            )
            return (1 - correlation).item()
        
        elif distance_metric == 'cosine':
            dot_product = torch.sum(exp1_flat * exp2_flat)
            norm1 = torch.sqrt(torch.sum(exp1_flat ** 2))
            norm2 = torch.sqrt(torch.sum(exp2_flat ** 2))
            cosine_sim = dot_product / (norm1 * norm2 + 1e-8)
            return (1 - cosine_sim).item()
        
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    def evaluate(
        self, 
        image: torch.Tensor, 
        target_class: int,
        n_samples: int = 50,
        noise_std: float = 0.1,
        noise_type: Literal['gaussian', 'uniform'] = 'gaussian',
        distance_metric: Literal['l1', 'l2', 'correlation', 'cosine'] = 'l1',
    ) -> float:
        """
        Compute Average Sensitivity: mean explanation change over perturbations.
        
        Args:
            image: Input image tensor of shape (1, C, H, W)
            target_class: Target class index (int)
            n_samples: Number of perturbed samples to generate
            noise_std: Standard deviation of noise (perturbation radius)
            noise_type: Type of noise to add - 'gaussian' or 'uniform'
            distance_metric: Metric to measure explanation distance
        
        Returns:
            avg_sensitivity: Mean explanation distance (float)
        """
        exp_original = self.explanation_method(image, target_class)
        
        distances_tensor = torch.zeros(n_samples, dtype=torch.float32)
        
        for i in tqdm(range(n_samples), desc="Computing sensitivity"):
            perturbed_image = self._add_noise(image, noise_std, noise_type)
            
            exp_perturbed = self.explanation_method(perturbed_image, target_class)
            
            distance = self._compute_explanation_distance(
                exp_original, exp_perturbed, distance_metric
            )
            
            distances_tensor[i] = distance
        
        avg_sensitivity = torch.mean(distances_tensor).item()
        
        return avg_sensitivity
    
    def evaluate_across_noise_levels(
        self,
        image: torch.Tensor,
        target_class: int,
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.15, 0.2],
        n_samples: int = 30,
        distance_metric: Literal['l1', 'l2', 'correlation', 'cosine'] = 'l1'
    ) -> Dict[str, List[float]]:
        """
        Evaluate sensitivity across different noise levels.
        
        Args:
            image: Input image tensor of shape (1, C, H, W)
            target_class: Target class index (int)
            noise_levels: List of noise standard deviations to test
            n_samples: Number of samples per noise level
            distance_metric: Distance metric to use
        
        Returns:
            Dictionary containing:
                - 'noise_levels': List of noise levels tested
                - 'sensitivities': List of average sensitivities at each noise level
        """
        sensitivities = []
        
        for noise_std in noise_levels:
            print(f"\nEvaluating at noise level {noise_std:.3f}...")
            sensitivity = self.evaluate(
                image, target_class, n_samples, noise_std, 
                distance_metric=distance_metric
            )
            sensitivities.append(sensitivity)
            print(f"Average Sensitivity: {sensitivity:.6f}")
        
        return {
            'noise_levels': noise_levels,
            'sensitivities': sensitivities
        }
    
    def plot_noise_level_analysis(
        self, 
        results: Dict[str, List[float]], 
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot sensitivity vs noise level.
        
        Args:
            results: Results dictionary from evaluate_across_noise_levels containing
                    'noise_levels' and 'sensitivities' keys
            save_path: Optional path to save the plot
        
        Returns:
            None (displays plot)
        """
        # Convert to torch tensors for max computation
        sensitivities_tensor = torch.tensor(results['sensitivities'], dtype=torch.float32)
        max_sensitivity = torch.max(sensitivities_tensor).item()
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(results['noise_levels'], results['sensitivities'], 
                'o-', linewidth=2, markersize=8, color='steelblue')
        
        plt.xlabel('Noise Level (Standard Deviation)', fontsize=12)
        plt.ylabel('Average Sensitivity', fontsize=12)
        plt.title('Explanation Robustness vs Noise Level', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add values on points
        for x, y in zip(results['noise_levels'], results['sensitivities']):
            plt.text(x, y + 0.01 * max_sensitivity, 
                    f'{y:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()