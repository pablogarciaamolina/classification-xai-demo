import torch

class AlexNet(torch.nn.Module):
    """
    Class for the AlexNet architecture.
    """
    def __init__(self, input_channels: int, num_classes=2):
        """
        Args:
            input_channels: Channels in the input image
            num_classes: Number of classes to classify
        """

        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(1, -1),

            torch.nn.LazyLinear(4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),

            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),

            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        """
        Forward method

        Args:
            x: Input of shape [batch, input_channels, h, w]

        Returns:
            Predictions of shape [batch, num_classes]
        """
        
        x = self.features(x)
        x = self.classifier(x)
        return x
