"""
XAI Service for generating explanations and evaluations
"""

import io
import base64
import torch
from PIL import Image
import numpy as np

from src.core.data._config import STL10_IMAGE_SIZE
from src.core.analysis.xai.methods.cam import Grad_CAM
from src.core.analysis.xai.methods.integrated_gradients import Integrated_Gradients
from src.core.analysis.xai.methods.gradient_ascent import Gradient_Ascent
from src.core.analysis.xai.evaluations.road import ROAD
from src.core.analysis.xai.evaluations.average_sensitivity import Average_Sensitivity

from app.backend.utils._utils import create_overlay
from app.backend.config import GRADCAM_TARGET_LAYER, ROAD_PERCENTILES, AVERAGE_SENSITIVITY_SAMPLES


class XAI_Service:
    """
    Wrapper for the XAI operations needed.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """        
        Args:
            model: PyTorch model to explain
            device: Device to run computations on
        """

        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.gradcam = Grad_CAM(model, GRADCAM_TARGET_LAYER)
        self.integrated_gradients = Integrated_Gradients(model)
        
        self.road = ROAD(model, device=device)
    
    def generate_local_explanation(
        self, 
        image: torch.Tensor, 
        target_class: int
    ) -> dict[str, any]:
        """
        Generate local explanation with saliency maps and evaluations
        
        Args:
            image: Input image tensor of shape [1, C, H, W]
            target_class: Target class index
            
        Returns:
            Dictionary containing saliency maps and evaluation metrics
        """

        image = image.to(self.device)
        
        gradcam_map = self.gradcam.explain(image, target_class)
        ig_map = self.integrated_gradients.explain(image, target_class)
        
        gradcam_road_results = self.road.evaluate(
            image, 
            gradcam_map, 
            target_class,
            percentiles=ROAD_PERCENTILES,
            metrics=['deletion'],
            verbose=False
        )
        gradcam_faithfulness = gradcam_road_results['deletion']['auc']
        
        ig_road_results = self.road.evaluate(
            image,
            ig_map,
            target_class,
            percentiles=ROAD_PERCENTILES,
            metrics=['deletion'],
            verbose=False
        )
        ig_faithfulness = ig_road_results['deletion']['auc']
        
        avg_sens_gradcam = Average_Sensitivity(self.gradcam.explain, device=self.device)
        gradcam_robustness_results = avg_sens_gradcam.evaluate_across_noise_levels(
            image,
            target_class,
            n_samples=AVERAGE_SENSITIVITY_SAMPLES,
            verbose=False
        )
        
        avg_sens_ig = Average_Sensitivity(self.integrated_gradients.explain, device=self.device)
        ig_robustness_results = avg_sens_ig.evaluate_across_noise_levels(
            image,
            target_class,
            n_samples=AVERAGE_SENSITIVITY_SAMPLES,
            verbose=False
        )
        
        gradcam_overlay = create_overlay(image, gradcam_map)
        ig_overlay = create_overlay(image, ig_map)
        
        return {
            'gradcam_map': gradcam_overlay,
            'integrated_gradients_map': ig_overlay,
            'gradcam_faithfulness': float(gradcam_faithfulness),
            'gradcam_robustness': np.mean(gradcam_robustness_results["sensitivities"]),
            'gradcam_deletion_curve': {
                'percentiles': gradcam_road_results['deletion']['percentiles'],
                'scores': gradcam_road_results['deletion']['scores']
            },
            'ig_faithfulness': float(ig_faithfulness),
            'ig_robustness': np.mean(ig_robustness_results["sensitivities"]),
            'ig_deletion_curve': {
                'percentiles': ig_road_results['deletion']['percentiles'],
                'scores': ig_road_results['deletion']['scores']
            }
        }
    
    def generate_global_explanation(self, target_class: int, class_name: str) -> dict[str, any]:
        """
        Generate global explanation using Gradient Ascent
        
        Args:
            target_class: Target class index
            class_name: Name of the target class
            
        Returns:
            Dictionary containing visualization and class name
        """

        grad_ascent = Gradient_Ascent(
            self.model,
            target_class,
            STL10_IMAGE_SIZE
        )
        
        generated_img = grad_ascent.generate(
            iterations=300,
            lr=0.1,
            l2_weight=0.001,
            tv_weight=0.01,
            blur_freq=10,
            verbose=False
        )
        
        img_display = generated_img.detach().cpu().squeeze().permute(1, 2, 0)
        img_display = torch.clamp(img_display * 255, 0, 255)
        img_np = img_display.numpy().astype('uint8')
        pil_img = Image.fromarray(img_np)
        
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            'visualization': img_base64,
            'class_name': class_name
        }
