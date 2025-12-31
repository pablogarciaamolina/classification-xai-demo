import argparse

import mlflow
from torch.utils.data.dataset import Subset

from src.core.data import load_big_cats, load_stl10, load_pediatric_pneumonia
from src.core.pipelines import Classifier_Pipeline, Pipeline_Config, load_model
from src.core.analysis import confusion_matrix, classification_metrics
from src.config import BATCH_SIZE, BIG_CATS_PIPELINE_CONFIG, STL10_PIPELINE_CONFIG, PEDIATRIC_PNEUMONIA_PIPELINE_CONFIG

DATA_LOADERS = {
    "big_cats": load_big_cats,
    "stl10": load_stl10,
    "pediatric_pneumonia": load_pediatric_pneumonia
}

PIPELINE_CONFIGS = {
    "big_cats": BIG_CATS_PIPELINE_CONFIG,
    "stl10": STL10_PIPELINE_CONFIG,
    "pediatric_pneumonia": PEDIATRIC_PNEUMONIA_PIPELINE_CONFIG
}

def evaluate(data_name: str, model_name: str) -> None:

    if data_name not in DATA_LOADERS:
        raise ValueError(f"Unknown dataset: {data_name}. Available: {list(DATA_LOADERS.keys())}")
    
    loader_func = DATA_LOADERS[data_name]
    pipeline_config_dict = PIPELINE_CONFIGS[data_name]

    print(f"Loading data for {data_name}...", end="")
    _, _, test_data = loader_func(batch_size=BATCH_SIZE, for_training=False)
    print("DONE")

    model, actual_model_name = load_model(model_name)
    print("Using model:", actual_model_name)

    config = Pipeline_Config(**pipeline_config_dict)
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

        metrics_fig = classification_metrics(
            predicted_labels,
            true_labels,
            save_name=f"{pipeline.name}_classification_metrics",
            labels=test_data.dataset.dataset.classes if type(test_data.dataset) == Subset else test_data.dataset.classes
        )

        mlflow.log_metrics(
            {
                "avg_test_accuracy": test_accuracy
            }
        )
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        mlflow.log_figure(metrics_fig, "classification_metrics.png")
            
        print("Evaluation registered with MLflow")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a classification model.")
    parser.add_argument("--data", type=str, required=True, help="Dataset to use (big_cats, stl10, pediatric_pneumonia)")
    parser.add_argument("--model-name", type=str, required=False, default=None, help="Name of the model to load.")
    
    args = parser.parse_args()
    evaluate(args.data, args.model_name)
