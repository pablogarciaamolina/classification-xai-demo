from flask import Blueprint, request, jsonify, current_app
from prometheus_client import Counter
import torch

from src.core.pipelines import Classifier_Pipeline

PREDICTION_COUNTER = Counter(
    'prediction_count',
    'Prediction counter for classification',
    ['prediction', 'correct']
)

predict_bp = Blueprint("predict_bp", __name__)

@predict_bp.route("/predict", methods=["POST"])
def predict() -> None:
    """
    Endpoint for the forwarding of predictions through the configurated pipeline

    Request form: json
        {
        "tensor": The unbatched input for the model,
        "true_label_idx": The true label index for the image (optional)
        }
    """
    try:
        pipeline: Classifier_Pipeline = current_app.config.get("pipeline")
        dataset_config = current_app.config.get("dataset_config")
        
        if pipeline is None or dataset_config is None:
            return jsonify({
                "error": "No dataset loaded. Please select a dataset first."
            }), 400
        
        class_mapping = dataset_config["classes"]
        
        data = request.get_json()
        t = torch.tensor(data["tensor"]).unsqueeze(0)
        prediction = pipeline.predict(t)

        PREDICTION_COUNTER.labels(
            prediction=class_mapping[prediction.item()],
            correct=(class_mapping[prediction.item()] == class_mapping[data["true_label_idx"]]) if "true_label_idx" in data else "N/A"
        ).inc()

        return jsonify({
            "predicted_class": prediction.item(),
        })

    except Exception as e:

        return jsonify({"error": str(e)}), 400
