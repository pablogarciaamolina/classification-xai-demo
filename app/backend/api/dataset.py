from flask import Blueprint, request, jsonify, current_app
from src.core.pipelines import load_model, Classifier_Pipeline, Pipeline_Config
from src.core.models import ResNet34
from app.backend.config import DATASET_CONFIGS
from app.backend.utils.xai_service import XAI_Service

dataset_bp = Blueprint("dataset_bp", __name__)

@dataset_bp.route("/dataset/select", methods=["POST"])
def select_dataset():
    """
    Endpoint for selecting and loading a dataset
    
    Request form: json
        {
        "dataset_name": The name of the dataset to load ("stl10" or "big_cats")
        }
    
    Returns:
        Dataset metadata including classes, image size, etc.
    """
    try:
        data = request.get_json()
        dataset_name = data.get("dataset_name", "").lower()
        
        if dataset_name not in DATASET_CONFIGS:
            return jsonify({
                "error": f"Invalid dataset name. Must be one of: {list(DATASET_CONFIGS.keys())}"
            }), 400
        
        config = DATASET_CONFIGS[dataset_name]
        
        print(f"🔄 Loading model for {dataset_name} dataset...")
        model, model_name = load_model(config["model_name"])
        print(f"✅ Model ({model_name}) loaded")
        
        print("🔄 Setting up pipeline...")
        pipeline = Classifier_Pipeline(model, config=Pipeline_Config(**config["pipeline_config"]))
        current_app.config["pipeline"] = pipeline
        print("✅ Pipeline set up successfully")
        
        print("🔄 Loading separate model for XAI service...")
        model_xai = ResNet34(
            in_channels=config["in_channels"],
            num_classes=config["num_classes"],
            small_inputs=config["small_inputs"]
        )
        model_xai.load_state_dict(model.state_dict())
        print("✅ Separate model loaded for XAI")
        
        print("🔄 Initializing XAI service...")
        xai_service = XAI_Service(model_xai, image_size=config["image_size"])
        current_app.config["xai_service"] = xai_service
        print("✅ XAI service initialized")
        
        current_app.config["current_dataset"] = dataset_name
        current_app.config["dataset_config"] = config
        
        return jsonify({
            "success": True,
            "dataset_name": dataset_name,
            "classes": config["classes"],
            "num_classes": config["num_classes"],
            "image_size": config["image_size"]
        })
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return jsonify({"error": str(e)}), 400

@dataset_bp.route("/dataset/current", methods=["GET"])
def get_current_dataset():
    """
    Endpoint for getting the currently loaded dataset information
    
    Returns:
        Current dataset metadata
    """
    try:
        dataset_name = current_app.config.get("current_dataset")
        
        if not dataset_name:
            return jsonify({
                "success": False,
                "message": "No dataset currently loaded"
            }), 404
        
        config = current_app.config.get("dataset_config", DATASET_CONFIGS.get(dataset_name, {}))
        
        return jsonify({
            "success": True,
            "dataset_name": dataset_name,
            "classes": config.get("classes", []),
            "num_classes": config.get("num_classes", 0),
            "image_size": config.get("image_size", [0, 0])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400
