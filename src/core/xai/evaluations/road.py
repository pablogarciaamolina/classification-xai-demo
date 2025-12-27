import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import Tuple, List, Dict, Union, Optional, Literal


class ROADEvaluation:
    """
    ROAD (Remove And Debias) - Evaluation metric for saliency maps.
    
    Measures the quality of explanations by removing pixels in order of importance
    and measuring the impact on model predictions.
    
    Reference: Rong et al. "A Consistent and Efficient Evaluation Strategy for 
    Attribution Methods" (ICML 2022)
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """
        Initialize ROAD evaluation.
        
        Args:
            model: PyTorch model to evaluate
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.model = model
        self.model.eval()
        self.device = device
        self.model.to(device)
    
    def _get_baseline(
        self, 
        image: torch.Tensor, 
        method: Literal['blur', 'mean', 'zero', 'noise'] = 'blur'
    ) -> torch.Tensor:
        """
        Get baseline replacement value for removed pixels.
        
        Args:
            image: Input image tensor of shape (1, C, H, W)
            method: Baseline method - 'blur', 'mean', 'zero', or 'noise'
        
        Returns:
            Baseline tensor with same shape as image (1, C, H, W)
        """
        if method == 'blur':
            sigma = 5.0
            
            img_np = image.cpu().numpy()[0].transpose(1, 2, 0)
            blurred = np.stack([
                gaussian_filter(img_np[:, :, i], sigma=sigma)
                for i in range(img_np.shape[2])
            ], axis=2)
            baseline = torch.from_numpy(blurred.transpose(2, 0, 1)).unsqueeze(0)
            return baseline.to(image.device)
        
        elif method == 'mean':
            return torch.mean(image, dim=(2, 3), keepdim=True).expand_as(image)
        
        elif method == 'zero':
            return torch.zeros_like(image)
        
        elif method == 'noise':
            return torch.randn_like(image) * 0.1 + 0.5
        
        else:
            raise ValueError(f"Unknown baseline method: {method}")
    
    def _normalize_saliency_map(
        self, 
        saliency_map: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Normalize saliency map to torch.Tensor with shape (1, 1, H, W).
        
        Args:
            saliency_map: Saliency map as numpy array or torch tensor
                         Can be shape (H, W), (1, H, W), (C, H, W), or (1, C, H, W)
        
        Returns:
            Normalized saliency map tensor of shape (1, 1, H, W)
        """
        # Convert to tensor if numpy array
        if isinstance(saliency_map, np.ndarray):
            saliency_map = torch.from_numpy(saliency_map).float()
        
        # Ensure it's a tensor
        if not isinstance(saliency_map, torch.Tensor):
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(saliency_map)}")
        
        # Normalize shape to (1, 1, H, W)
        if len(saliency_map.shape) == 2:
            # (H, W) -> (1, 1, H, W)
            saliency_map = saliency_map.unsqueeze(0).unsqueeze(0)
        elif len(saliency_map.shape) == 3:
            # (1, H, W) or (C, H, W) -> (1, 1, H, W)
            # Assume first dimension is batch or channel, take mean if multiple channels
            if saliency_map.shape[0] > 1:
                saliency_map = saliency_map.mean(dim=0, keepdim=True).unsqueeze(0)
            else:
                saliency_map = saliency_map.unsqueeze(0)
        elif len(saliency_map.shape) == 4:
            # (1, C, H, W) -> (1, 1, H, W)
            if saliency_map.shape[1] > 1:
                saliency_map = saliency_map.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unexpected saliency map shape: {saliency_map.shape}")
        
        return saliency_map
    
    def _remove_pixels(
        self, 
        image: torch.Tensor, 
        saliency_map: torch.Tensor, 
        percentile: float, 
        baseline: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Remove top percentile of most important pixels according to saliency map.
        
        Args:
            image: Input image of shape (1, C, H, W)
            saliency_map: Saliency map of shape (1, 1, H, W)
            percentile: Percentage of pixels to remove (0-100)
            baseline: Baseline image to use for replacement, shape (1, C, H, W)
        
        Returns:
            Tuple of:
                - Modified image with top pixels replaced by baseline (1, C, H, W)
                - Binary mask indicating which pixels were kept (1, 1, H, W)
        """
        # Normalize saliency map
        saliency_map = self._normalize_saliency_map(saliency_map)
        
        # Flatten for quantile computation
        sal_flat = saliency_map.view(-1)
        
        # Compute threshold
        threshold = torch.quantile(sal_flat, 1.0 - percentile / 100.0)
        
        # Create mask (1 = keep, 0 = remove)
        mask = (saliency_map <= threshold).float()
        
        # Apply mask
        modified_image = image * mask + baseline * (1 - mask)
        
        return modified_image, mask
    
    def compute_deletion_curve(
        self, 
        image: torch.Tensor, 
        saliency_map: Union[np.ndarray, torch.Tensor], 
        target_class: int,
        percentiles: Optional[List[int]] = None,
        baseline_method: Literal['blur', 'mean', 'zero', 'noise'] = 'blur'
    ) -> Tuple[List[int], List[float]]:
        """
        Compute deletion curve: model confidence as pixels are removed.
        
        Args:
            image: Input image tensor of shape (1, C, H, W)
            saliency_map: Saliency map as numpy array or torch tensor
                         Can be shape (H, W), (1, H, W), (C, H, W), or (1, C, H, W)
            target_class: Target class index (int)
            percentiles: List of percentiles to evaluate (default: 0 to 100 in steps of 5)
            baseline_method: Method for baseline replacement
        
        Returns:
            Tuple of:
                - percentiles: List of percentile values
                - scores: List of model confidence scores at each percentile
        """
        if percentiles is None:
            percentiles = list(range(0, 101, 5))
        
        # Normalize saliency map
        saliency_map = self._normalize_saliency_map(saliency_map)
        saliency_map = saliency_map.to(image.device)
        
        # Get baseline
        baseline = self._get_baseline(image, method=baseline_method)
        
        scores = []
        
        for p in percentiles:
            # Remove top p% of pixels
            modified_image, _ = self._remove_pixels(image, saliency_map, p, baseline)
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(modified_image.to(self.device))
                prob = torch.softmax(output, dim=1)
                score = prob[0, target_class].item()
            
            scores.append(score)
        
        return percentiles, scores
    
    def compute_insertion_curve(
        self, 
        image: torch.Tensor, 
        saliency_map: Union[np.ndarray, torch.Tensor], 
        target_class: int,
        percentiles: Optional[List[int]] = None,
        baseline_method: Literal['blur', 'mean', 'zero', 'noise'] = 'blur'
    ) -> Tuple[List[int], List[float]]:
        """
        Compute insertion curve: model confidence as pixels are added back.
        
        Args:
            image: Input image tensor of shape (1, C, H, W)
            saliency_map: Saliency map as numpy array or torch tensor
                         Can be shape (H, W), (1, H, W), (C, H, W), or (1, C, H, W)
            target_class: Target class index (int)
            percentiles: List of percentiles to evaluate (default: 0 to 100 in steps of 5)
            baseline_method: Method for baseline replacement
        
        Returns:
            Tuple of:
                - percentiles: List of percentile values
                - scores: List of model confidence scores at each percentile
        """
        if percentiles is None:
            percentiles = list(range(0, 101, 5))
        
        # Normalize saliency map
        saliency_map = self._normalize_saliency_map(saliency_map)
        saliency_map = saliency_map.to(image.device)
        
        # Get baseline
        baseline = self._get_baseline(image, method=baseline_method)
        
        scores = []
        
        for p in percentiles:
            # Compute threshold for insertion
            sal_flat = saliency_map.view(-1)
            threshold = torch.quantile(sal_flat, 1.0 - p / 100.0)
            
            # Create mask (1 = insert, 0 = baseline)
            mask = (saliency_map >= threshold).float()
            
            # Apply mask (opposite of deletion)
            modified_image = image * mask + baseline * (1 - mask)
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(modified_image.to(self.device))
                prob = torch.softmax(output, dim=1)
                score = prob[0, target_class].item()
            
            scores.append(score)
        
        return percentiles, scores
    
    def _compute_auc(
        self, 
        percentiles: List[int], 
        scores: List[float]
    ) -> float:
        """
        Compute Area Under Curve using trapezoidal rule.
        
        Args:
            percentiles: List of x values (percentiles)
            scores: List of y values (confidence scores)
        
        Returns:
            AUC value normalized to [0, 1]
        """
        return np.trapz(scores, percentiles) / 100.0
    
    def evaluate(
        self, 
        image: torch.Tensor, 
        saliency_map: Union[np.ndarray, torch.Tensor], 
        target_class: int,
        percentiles: Optional[List[int]] = None,
        baseline_method: Literal['blur', 'mean', 'zero', 'noise'] = 'blur',
        metrics: List[Literal['deletion', 'insertion']] = ['deletion', 'insertion']
    ) -> Dict[str, Dict[str, Union[List[int], List[float], float]]]:
        """
        Complete ROAD evaluation of a saliency map.
        
        Args:
            image: Input image tensor of shape (1, C, H, W)
            saliency_map: Saliency map as numpy array or torch tensor
                         Can be shape (H, W), (1, H, W), (C, H, W), or (1, C, H, W)
            target_class: Target class index (int)
            percentiles: List of percentiles to evaluate (default: 0 to 100 in steps of 5)
            baseline_method: Method for baseline replacement
            metrics: List of metrics to compute ('deletion', 'insertion')
        
        Returns:
            Dictionary with evaluation results containing:
                - 'deletion': {'percentiles': List[int], 'scores': List[float], 'auc': float}
                - 'insertion': {'percentiles': List[int], 'scores': List[float], 'auc': float}
        """
        results = {}
        
        if 'deletion' in metrics:
            print("Computing deletion curve...")
            del_percentiles, del_scores = self.compute_deletion_curve(
                image, saliency_map, target_class, percentiles, baseline_method
            )
            del_auc = self._compute_auc(del_percentiles, del_scores)
            
            results['deletion'] = {
                'percentiles': del_percentiles,
                'scores': del_scores,
                'auc': del_auc
            }
            print(f"Deletion AUC: {del_auc:.4f}")
        
        if 'insertion' in metrics:
            print("Computing insertion curve...")
            ins_percentiles, ins_scores = self.compute_insertion_curve(
                image, saliency_map, target_class, percentiles, baseline_method
            )
            ins_auc = self._compute_auc(ins_percentiles, ins_scores)
            
            results['insertion'] = {
                'percentiles': ins_percentiles,
                'scores': ins_scores,
                'auc': ins_auc
            }
            print(f"Insertion AUC: {ins_auc:.4f}")
        
        return results
    
    def plot_curves(
        self, 
        results: Dict[str, Dict[str, Union[List[int], List[float], float]]], 
        title: str = "ROAD Evaluation", 
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot deletion and insertion curves.
        
        Args:
            results: Results dictionary from evaluate()
            title: Plot title
            save_path: Path to save the plot (optional)
        
        Returns:
            None (displays plot)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        if 'deletion' in results:
            ax = axes[0]
            del_data = results['deletion']
            ax.plot(del_data['percentiles'], del_data['scores'], 'r-', linewidth=2, label='Deletion')
            ax.set_xlabel('Percentage of Pixels Removed (%)', fontsize=11)
            ax.set_ylabel('Model Confidence', fontsize=11)
            ax.set_title(f"Deletion Curve (AUC: {del_data['auc']:.4f})", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 1])
        
        if 'insertion' in results:
            ax = axes[1]
            ins_data = results['insertion']
            ax.plot(ins_data['percentiles'], ins_data['scores'], 'g-', linewidth=2, label='Insertion')
            ax.set_xlabel('Percentage of Pixels Inserted (%)', fontsize=11)
            ax.set_ylabel('Model Confidence', fontsize=11)
            ax.set_title(f"Insertion Curve (AUC: {ins_data['auc']:.4f})", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 1])
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def visualize_removal_steps(
        self, 
        image: torch.Tensor, 
        saliency_map: Union[np.ndarray, torch.Tensor], 
        percentiles: List[int] = [10, 30, 50, 70, 90],
        baseline_method: Literal['blur', 'mean', 'zero', 'noise'] = 'blur'
    ) -> None:
        """
        Visualize the image at different removal percentiles.
        
        Args:
            image: Input image tensor of shape (1, C, H, W)
            saliency_map: Saliency map as numpy array or torch tensor
            percentiles: List of percentiles to visualize
            baseline_method: Baseline replacement method
        
        Returns:
            None (displays plot)
        """
        # Normalize saliency map
        saliency_map = self._normalize_saliency_map(saliency_map)
        saliency_map = saliency_map.to(image.device)
        
        baseline = self._get_baseline(image, method=baseline_method)
        
        n = len(percentiles) + 1
        fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
        
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[0].imshow(img_np)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        for idx, p in enumerate(percentiles):
            modified_image, mask = self._remove_pixels(image, saliency_map, p, baseline)
            img_np = modified_image[0].cpu().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            
            axes[idx + 1].imshow(img_np)
            axes[idx + 1].set_title(f'{p}% Removed')
            axes[idx + 1].axis('off')
        
        plt.tight_layout()
        plt.show()