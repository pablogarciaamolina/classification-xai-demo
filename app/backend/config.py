import os

BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))
MODEL_NAME = os.getenv("MODEL_NAME", "pretrained/STL10_ResNet34_base_model")

GRADCAM_TARGET_LAYER = "processing_block.15.conv_branch.3"
ROAD_PERCENTILES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
AVERAGE_SENSITIVITY_SAMPLES = 30