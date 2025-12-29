import argparse

import mlflow
from torch.utils.data.dataset import Subset

from src.core.data import load_big_cats
from src.core.pipelines import Classifier_Pipeline, Pipeline_Config, load_model
from src.core.analysis import confusion_matrix
from src.config import BATCH_SIZE, BIG_CATS_PIPELINE_CONFIG


def evaluate(model_name: str) -> None:

    print("Loading data...", end="")
    _, _, test_data = load_big_cats(batch_size=BATCH_SIZE, for_training=False)
    print("DONE")

    model, actual_model_name = load_model(model_name)
    print("Using model:", actual_model_name)

    config = Pipeline_Config(**BIG_CATS_PIPELINE_CONFIG)
    print("Using device:", config.device)
    pipeline = Classifier_Pipeline(model, config=config, name=actual_model_name)
    
    mlflow.set_experiment(pipeline.name)
    with mlflow.start_run():
        test_accuracy, predicted_labels, true_labels = pipeline.evaluate(
            test_data
        )
        print("Test accuracy:", test_accuracy)

        cm_fig = confusion_matrix(
            predicted_labels,
            true_labels,
            save_name=f"{pipeline.name}_confusion_matrix",
            labels=test_data.dataset.dataset.classes if type(test_data.dataset) == Subset else test_data.dataset.classes
        )

        mlflow.log_metrics(
            {
                "avg_test_accuracy": test_accuracy
            }
        )
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        print("Evaluation registered with MLflow")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default=None,
        help="Name of the model to load (must match saved model name)."
    )

    args = parser.parse_args()
    evaluate(args.model_name)
