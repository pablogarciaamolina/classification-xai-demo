from flask import Blueprint, request, jsonify, current_app
from prometheus_client import Counter
import torch

from src.core.pipelines import Classifier_Pipeline
from src.core.data.stl10 import STL10_Dataset

PREDICTION_COUNTER = Counter(
    'stl10_prediction_count',
    'Prediction counter for stl10 classification',
    ['prediction', 'correct']
)
CLASS_MAPPING = STL10_Dataset.classes

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

        pipeline: Classifier_Pipeline = current_app.config["pipeline"]

        data = request.get_json()
        t = torch.tensor(data["tensor"]).unsqueeze(0)
        prediction = pipeline.predict(t)

        PREDICTION_COUNTER.labels(
            prediction=CLASS_MAPPING[prediction.item()],
            correct=(CLASS_MAPPING[prediction.item()] == CLASS_MAPPING[data["true_label_idx"]]) if "true_label_idx" in data else "N/A"
        ).inc()

        return jsonify({
            "predicted_class": prediction.item(),
        })

    except Exception as e:

        return jsonify({"error": str(e)}), 400
