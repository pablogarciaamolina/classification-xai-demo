import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.core.pipelines._config import METRICS_DIR
from src.core.pipelines._base import Pipeline_Config
from src.core.pipelines._utils import classifier_train_step, classifier_val_step, classifier_test_step, save_model

class Classifier_Pipeline:
    """
    Class implementing the customizable pipeline for a classifier model
    """

    def __init__(self, model: torch.nn.Module, config: Pipeline_Config, name: str = None):
        """
        Constructor for the pipeline.

        Args:
            model: The classifier module.
            config: The configuration for the pipeline.
            name: An optional name for the pipeline. If not provided, a name will be automatically generated.
        """

        self.model = model.to(config.device)
        self.config = config
        self.name = name if name else f"{self.model.__class__.__name__}_{time.time()}"

    def train(self,
        train_data: DataLoader,
        val_data: DataLoader,
        add_to_name: str = "",
        save_model: bool = True
    ) -> tuple[float]:
        """
        Method for training the model

        Args:
            train_data: Training data.
            val_data: Validation data.
            add_to_name: An optional string to add to the end of the automatically constructed name, used when saving or identifying the model.
            save_model: Whether to sve the model at the end of training. Defaults to `True`.

        Returns:
            A tuple containing the average training accuracy and the average validation accuracy.
        """
        
        self.name += add_to_name
        writer: SummaryWriter = SummaryWriter(os.path.join(METRICS_DIR, self.name))

        model: torch.Module = self.model.to(self.config.device)

        loss = self.config.loss_class()
        optimizer: torch.optim.Optimizer = self.config.optimizer_class(
            model.parameters(), **self.config.optimizer_kwargs
        )
        scheduler = self.config.scheduler_class(
            optimizer, **self.config.scheduler_kwargs
        )

        with tqdm(range(self.config.epochs)) as t:

            for epoch in t:

                train_mean_accuracy = classifier_train_step(
                    model, train_data, loss, optimizer, epoch, self.config.device, writer
                )

                val_mean_accuracy = classifier_val_step(
                    model, val_data, loss, epoch, self.config.device, writer
                )

                t.set_postfix({
                    "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                    "train accuracy": f"{round(train_mean_accuracy, 4)}",
                    "val accuracy": f"{round(val_mean_accuracy, 4)}"
                })

                scheduler.step()

        if save_model:
            save_model(model, self.name)

        return (train_mean_accuracy, val_mean_accuracy,)
    
    def evaluate(
        self,
        test_data: DataLoader,
    ) -> tuple[float, list[int], list[int]]:
        """
        Method for evaluating the model.

        Args:
            test_data: The data to evaluate the model on.

        Returns:
            Tuple of the mean accuracy over testing, predicted labels and true labels.
        """
        
        model: torch.Module = self.model.to(self.config.device)
        accuracy, predicted_labels, true_labels = classifier_test_step(model, test_data, self.config.device)

        return accuracy, predicted_labels, true_labels
    
    def predict(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Method for quickly predicting a single input.

        Args:
            input: The input data for the model. THe input must be batched. See model.forward() for more information on the shape of the input.

        Returns:
            The predictions of the model of shape [batch].
        """
        
        return torch.argmax(self.model(input.to(self.config.device)), dim=1)