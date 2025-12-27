import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



class ROADEvaluation:
    """
    ROAD (Remove And Debias) - Evaluation metric for saliency maps.
    
    Measures the quality of explanations by removing pixels in order of importance
    and measuring the impact on model predictions.
    
    Reference: Rong et al. "A Consistent and Efficient Evaluation Strategy for 
    Attribution Methods" (ICML 2022)
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: PyTorch model to evaluate
            device: Device to run computations on
        """
        self.model = model
        self.model.eval()
        self.device = device
        self.model.to(device)
    
    def get_baseline(self, image, method='blur'):
        """
        Get baseline replacement value for removed pixels.
        
        Args:
            image: Input image tensor (1, C, H, W)
            method: 'blur', 'mean', 'zero', or 'noise'
        
        Returns:
            Baseline tensor with same shape as image
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
    
    def remove_pixels(self, image, saliency_map, percentile, baseline):
        """
        Remove top percentile of most important pixels according to saliency map.
        
        Args:
            image: Input image (1, C, H, W)
            saliency_map: Saliency map (1, 1, H, W) or (H, W)
            percentile: Percentage of pixels to remove (0-100)
            baseline: Baseline image to use for replacement
        
        Returns:
            Modified image with top pixels replaced by baseline
        """
        if len(saliency_map.shape) == 2:
            saliency_map = saliency_map.unsqueeze(0).unsqueeze(0)
        elif len(saliency_map.shape) == 3:
            saliency_map = saliency_map.unsqueeze(0)
        
        sal_flat = saliency_map.view(-1)
        
        threshold = torch.quantile(sal_flat, 1.0 - percentile / 100.0)
        
        mask = (saliency_map <= threshold).float()
        
        modified_image = image * mask + baseline * (1 - mask)
        
        return modified_image, mask
    
    def compute_deletion_curve(
        self, 
        image, 
        saliency_map, 
        target_class,
        percentiles=None,
        baseline_method='blur'
    ):
        """
        Compute deletion curve: model confidence as pixels are removed.
        
        Args:
            image: Input image (1, C, H, W)
            saliency_map: Saliency map (H, W) or (1, 1, H, W)
            target_class: Target class index
            percentiles: List of percentiles to evaluate (default: 0 to 100 in steps of 5)
            baseline_method: Method for baseline replacement
        
        Returns:
            percentiles: List of percentile values
            scores: List of model scores at each percentile
        """
        if percentiles is None:
            percentiles = list(range(0, 101, 5))
        
        # Get baseline
        baseline = self.get_baseline(image, method=baseline_method)
        
        scores = []
        
        for p in percentiles:
            # Remove top p% of pixels
            modified_image, _ = self.remove_pixels(image, saliency_map, p, baseline)
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(modified_image.to(self.device))
                prob = torch.softmax(output, dim=1)
                score = prob[0, target_class].item()
            
            scores.append(score)
        
        return percentiles, scores
    
    def compute_insertion_curve(
        self, 
        image, 
        saliency_map, 
        target_class,
        percentiles=None,
        baseline_method='blur'
    ):
        """
        Compute insertion curve: model confidence as pixels are added back.
        
        Args:
            image: Input image (1, C, H, W)
            saliency_map: Saliency map (H, W) or (1, 1, H, W)
            target_class: Target class index
            percentiles: List of percentiles to evaluate
            baseline_method: Method for baseline replacement
        
        Returns:
            percentiles: List of percentile values
            scores: List of model scores at each percentile
        """

        if percentiles is None:
            percentiles = list(range(0, 101, 5))
        
        baseline = self.get_baseline(image, method=baseline_method)
        
        if len(saliency_map.shape) == 2:
            saliency_map_t = torch.from_numpy(saliency_map).unsqueeze(0).unsqueeze(0)
        elif len(saliency_map.shape) == 3:
            saliency_map_t = torch.from_numpy(saliency_map).unsqueeze(0) if isinstance(saliency_map, np.ndarray) else saliency_map.unsqueeze(0)
        else:
            saliency_map_t = torch.from_numpy(saliency_map) if isinstance(saliency_map, np.ndarray) else saliency_map
        
        scores = []
        
        for p in percentiles:
            sal_flat = saliency_map_t.view(-1)
            threshold = torch.quantile(sal_flat, 1.0 - p / 100.0)
            
            mask = (saliency_map_t >= threshold).float()
            
            modified_image = image * mask + baseline * (1 - mask)
            
            with torch.no_grad():
                output = self.model(modified_image.to(self.device))
                prob = torch.softmax(output, dim=1)
                score = prob[0, target_class].item()
            
            scores.append(score)
        
        return percentiles, scores
    
    def compute_auc(self, percentiles, scores):
        """
        Compute Area Under Curve using trapezoidal rule.
        
        Args:
            percentiles: List of x values
            scores: List of y values
        
        Returns:
            AUC value
        """
        return np.trapz(scores, percentiles) / 100.0
    
    def evaluate(
        self, 
        image, 
        saliency_map, 
        target_class,
        percentiles=None,
        baseline_method='blur',
        metrics=['deletion', 'insertion']
    ):
        """
        Complete ROAD evaluation of a saliency map.
        
        Args:
            image: Input image (1, C, H, W)
            saliency_map: Saliency map (H, W) or (1, 1, H, W)
            target_class: Target class index
            percentiles: List of percentiles to evaluate
            baseline_method: Method for baseline replacement
            metrics: List of metrics to compute ('deletion', 'insertion')
        
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        if isinstance(saliency_map, np.ndarray):
            saliency_map = torch.from_numpy(saliency_map).float()
        
        saliency_map = saliency_map.to(image.device)
        
        if 'deletion' in metrics:
            print("Computing deletion curve...")
            del_percentiles, del_scores = self.compute_deletion_curve(
                image, saliency_map, target_class, percentiles, baseline_method
            )
            del_auc = self.compute_auc(del_percentiles, del_scores)
            
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
            ins_auc = self.compute_auc(ins_percentiles, ins_scores)
            
            results['insertion'] = {
                'percentiles': ins_percentiles,
                'scores': ins_scores,
                'auc': ins_auc
            }
            print(f"Insertion AUC: {ins_auc:.4f}")
        
        return results
    
    def plot_curves(self, results, title="ROAD Evaluation", save_path=None):
        """
        Plot deletion and insertion curves.
        
        Args:
            results: Results dictionary from evaluate()
            title: Plot title
            save_path: Path to save the plot (optional)
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
        image, 
        saliency_map, 
        percentiles=[10, 30, 50, 70, 90],
        baseline_method='blur'
    ):
        """
        Visualize the image at different removal percentiles.
        
        Args:
            image: Input image (1, C, H, W)
            saliency_map: Saliency map
            percentiles: List of percentiles to visualize
            baseline_method: Baseline replacement method
        """
        baseline = self.get_baseline(image, method=baseline_method)
        
        n = len(percentiles) + 1
        fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
        
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[0].imshow(img_np)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        for idx, p in enumerate(percentiles):
            modified_image, mask = self.remove_pixels(image, saliency_map, p, baseline)
            img_np = modified_image[0].cpu().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            
            axes[idx + 1].imshow(img_np)
            axes[idx + 1].set_title(f'{p}% Removed')
            axes[idx + 1].axis('off')
        
        plt.tight_layout()
        plt.show()