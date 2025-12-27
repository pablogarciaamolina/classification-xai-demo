"""
Backend main
"""

from flask import Flask
from .api.predict import predict_bp
from .api.metrics import metrics_bp
from .config import MODEL_NAME, BACKEND_PORT

from src.config import BIG_CATS_PIPELINE_CONFIG
from src.core.pipelines import load_model, Classifier_Pipeline, Pipeline_Config

def raise_backend() -> Flask:
    """
    Raises the backend for classification and analysis, based on the parameters set on the config file.
    """
    app = Flask(__name__)

    print("🔄 Loading model...")
    model, model_name = load_model(MODEL_NAME)
    print(f"✅ Model ({model_name}) loaded")

    print("🔄 Setting up pipeline...")
    pipeline = Classifier_Pipeline(model, config=Pipeline_Config(**BIG_CATS_PIPELINE_CONFIG))
    app.config["pipeline"] = pipeline
    print("✅ Pipeline set up successfully")

    # Endpoints
    app.register_blueprint(predict_bp)
    app.register_blueprint(metrics_bp)

    return app


if __name__ == "__main__":
    app = raise_backend()
    app.run(host="0.0.0.0", port=BACKEND_PORT)
