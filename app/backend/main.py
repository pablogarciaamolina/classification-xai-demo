"""
Backend main
"""

from flask import Flask
from .api.predict import predict_bp
from .api.metrics import metrics_bp
from .api.xai import xai_bp
from .config import MODEL_NAME, BACKEND_PORT

from src.config import BIG_CATS_PIPELINE_CONFIG
from src.core.pipelines import load_model, Classifier_Pipeline, Pipeline_Config
from src.core.models.alexnet import AlexNet
from .utils.xai_service import XAI_Service

def raise_backend() -> Flask:
    """
    Raises the backend for classification and analysis, based on the parameters set on the config file.
    """
    app = Flask(__name__)

    print("🔄 Loading model for prediction pipeline...")
    model, model_name = load_model(MODEL_NAME)
    print(f"✅ Model ({model_name}) loaded")

    print("🔄 Setting up pipeline...")
    pipeline = Classifier_Pipeline(model, config=Pipeline_Config(**BIG_CATS_PIPELINE_CONFIG))
    app.config["pipeline"] = pipeline
    print("✅ Pipeline set up successfully")

    print("🔄 Loading separate model for XAI service...")
    model_xai = AlexNet(input_channels=3, num_classes=10)
    model_xai.load_state_dict(model.state_dict())
    print("✅ Separate model loaded for XAI")

    print("🔄 Initializing XAI service...")
    xai_service = XAI_Service(model_xai, device=str(BIG_CATS_PIPELINE_CONFIG["device"]))
    app.config["xai_service"] = xai_service
    print("✅ XAI service initialized")

    # Endpoints
    app.register_blueprint(predict_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(xai_bp)

    return app


if __name__ == "__main__":
    app = raise_backend()
    app.run(host="0.0.0.0", port=BACKEND_PORT)

