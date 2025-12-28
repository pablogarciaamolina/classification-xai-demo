"""
XAI API endpoints for local and global explanations
"""

from flask import Blueprint, request, jsonify, current_app
import torch

from src.core.data.big_cats import Big_Cats_Dataset
from app.backend.utils.xai_service import XAI_Service

xai_bp = Blueprint("xai_bp", __name__)

CLASS_MAPPING = Big_Cats_Dataset.classes


@xai_bp.route("/xai/local", methods=["POST"])
def local_explanation():
    """
    Generate local explanation with saliency maps and evaluations
    
    Request form: json
        {
            "tensor": The unbatched input for the model,
            "target_class": The target class index for explanation
        }
    
    Returns:
        JSON with saliency maps (base64) and evaluation metrics
    """
    try:
        xai_service: XAI_Service = current_app.config["xai_service"]
        
        data = request.get_json()
        tensor = torch.tensor(data["tensor"]).unsqueeze(0)
        target_class = data["target_class"]
        
        result = xai_service.generate_local_explanation(tensor, target_class)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@xai_bp.route("/xai/global", methods=["POST"])
def global_explanation():
    """
    Generate global explanation using Gradient Ascent
    
    Request form: json
        {
            "target_class": The target class index
        }
    
    Returns:
        JSON with visualization (base64) and class name
    """
    try:
        xai_service: XAI_Service = current_app.config["xai_service"]
        
        data = request.get_json()
        target_class = data["target_class"]
        class_name = CLASS_MAPPING[target_class]
        
        result = xai_service.generate_global_explanation(target_class, class_name)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
