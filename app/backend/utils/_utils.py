import io
import base64
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_overlay(image: torch.Tensor, saliency_map: torch.Tensor, alpha: float = 0.5) -> str:
    """
    Create an overlay of saliency map on original image
    
    Args:
        image: Original image tensor of shape [1, C, H, W] or [C, H, W]
        saliency_map: Saliency map tensor of shape [H, W]
        alpha: Transparency of overlay
        
    Returns:
        Base64-encoded PNG image string
    """

    if image.dim() == 4:
        image = image.squeeze(0)
    
    img_np = image.cpu().detach().permute(1, 2, 0).numpy()
    
    img_np = (img_np * 0.5) + 0.5
    img_np = np.clip(img_np, 0, 1)
    
    saliency_np = saliency_map.cpu().detach().numpy()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.imshow(saliency_np, cmap='jet', alpha=alpha)
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64