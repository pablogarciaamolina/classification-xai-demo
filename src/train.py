import argparse

import torch
import mlflow
import mlflow.pytorch

from src.core.data import load_big_cats, load_stl10, load_pediatric_pneumonia
from src.core.models import AlexNet, ResNet34
from src.core.pipelines import Classifier_Pipeline, Pipeline_Config, save_model
from src.config import NUM_CLASSES, BATCH_SIZE, BIG_CATS_PIPELINE_CONFIG, STL10_PIPELINE_CONFIG, PEDIATRIC_PNEUMONIA_PIPELINE_CONFIG


DATA_LOADERS = {
    "big_cats": load_big_cats,
    "stl10": load_stl10,
    "pediatric_pneumonia": load_pediatric_pneumonia
}

MODEL_SMALL_INPUTS = {
    "big_cats": False,
    "stl10": True,
    "pediatric_pneumonia": False
}

PIPELINE_CONFIGS = {
    "big_cats": BIG_CATS_PIPELINE_CONFIG,
    "stl10": STL10_PIPELINE_CONFIG,
    "pediatric_pneumonia": PEDIATRIC_PNEUMONIA_PIPELINE_CONFIG
}

def train(data_name: str, model_name: str) -> None:
    if data_name not in DATA_LOADERS:
        raise ValueError(f"Unknown dataset: {data_name}. Available: {list(DATA_LOADERS.keys())}")
    
    loader_func = DATA_LOADERS[data_name]
    pipeline_config_dict = PIPELINE_CONFIGS[data_name]
    
    print(f"Loading data for {data_name}...", end="")
    train_data, val_data, _ = loader_func(batch_size=BATCH_SIZE, for_training=True)
    print("DONE")

    inputs: torch.Tensor = next(iter(train_data))[0]
    input_channels = inputs.shape[1]
    
    use_small_inputs = MODEL_SMALL_INPUTS.get(data_name, False)
    
    if model_name == "resnet":
        model = ResNet34(in_channels=input_channels, num_classes=NUM_CLASSES, small_inputs=use_small_inputs)
    elif model_name == "alexnet":
        model = AlexNet(input_channels=input_channels, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: ['resnet', 'alexnet']")
    
    config = Pipeline_Config(**pipeline_config_dict)
    print("Using device:", config.device)
    pipeline = Classifier_Pipeline(model, config=config)
    pipeline.name = f"{data_name}_{pipeline.name}"
    
    mlflow.set_experiment(pipeline.name)
    with mlflow.start_run():
        avg_train_acc, avg_val_acc = pipeline.train(
            train_data,
            val_data,
            save_model=False
        )
        save_model(model, pipeline.name)
    
        mlflow.pytorch.log_model(model, f"{data_name}-model", input_example=inputs.numpy())
        mlflow.log_params(
            {
                "num_classes": NUM_CLASSES,
                "batch_size": BATCH_SIZE,
                "dataset": data_name,
                "model_architecture": model_name,
                **pipeline_config_dict
            }
        )
        mlflow.log_metrics(
            {
                "avg_train_accuracy": avg_train_acc,
                "avg_validation_accuracy": avg_val_acc
            }
        )
        print("Experiment registered with MLflow")

    print(f"Model trained | train accuracy: {avg_train_acc:.4f}, validation accucary: {avg_val_acc:.4f}")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument("--data", type=str, required=True, help="Dataset to use (big_cats, stl10, pediatric_pneumonia)")
    parser.add_argument("--model", type=str, required=False, default="resnet", choices=["resnet", "alexnet"], help="Model architecture to use")
    args = parser.parse_args()
    
    train(args.data, args.model)
