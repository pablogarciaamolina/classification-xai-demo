"""
Backend main
"""

import matplotlib
matplotlib.use('Agg')
from flask import Flask
from .api.predict import predict_bp
from .api.metrics import metrics_bp
from .api.xai import xai_bp
from .api.dataset import dataset_bp
from .config import BACKEND_PORT

def raise_backend() -> Flask:
    """
    Raises the backend for classification and analysis.
    Dataset must be selected via the /dataset/select endpoint before use.
    """
    app = Flask(__name__)
    
    app.config["current_dataset"] = None
    app.config["dataset_config"] = None
    app.config["pipeline"] = None
    app.config["xai_service"] = None
    
    print("✅ Backend initialized (no dataset loaded)")
    print("📌 Use /dataset/select endpoint to load a dataset")

    app.register_blueprint(predict_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(xai_bp)
    app.register_blueprint(dataset_bp)

    return app


if __name__ == "__main__":
    app = raise_backend()
    app.run(host="0.0.0.0", port=BACKEND_PORT)

