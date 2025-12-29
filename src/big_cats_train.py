import torch

from src.core.data import load_big_cats
from src.core.models import AlexNet, ResNet34
from src.core.pipelines import Classifier_Pipeline, Pipeline_Config, save_model
from src.config import NUM_CLASSES, BATCH_SIZE, BIG_CATS_PIPELINE_CONFIG

import mlflow
import mlflow.pytorch

def train() -> None:

    print("Loading data...", end="")
    train_data, val_data, _ = load_big_cats(batch_size=BATCH_SIZE, for_training=True)
    print("DONE")

    inputs: torch.Tensor = next(iter(train_data))[0]
    # model = AlexNet(inputs.shape[1], NUM_CLASSES)
    model = ResNet34(in_channels=inputs.shape[1], num_classes=NUM_CLASSES, small_inputs=False)

    config = Pipeline_Config(**BIG_CATS_PIPELINE_CONFIG)
    print("Using device:", config.device)
    pipeline = Classifier_Pipeline(model, config=config)
    pipeline.name = "Big_Cats_" + pipeline.name
    
    mlflow.set_experiment(pipeline.name)
    with mlflow.start_run():
        avg_train_acc, avg_val_acc = pipeline.train(
            train_data,
            val_data,
            save_model=False
        )
        save_model(model, pipeline.name)
    
        mlflow.pytorch.log_model(model, "big-cats-model", input_example=inputs.numpy())
        mlflow.log_params(
            {
                "num_classes": NUM_CLASSES,
                "batch_size": BATCH_SIZE,
                **BIG_CATS_PIPELINE_CONFIG
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
    train()
