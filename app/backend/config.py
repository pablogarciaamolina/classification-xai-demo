import os

BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))
MODEL_NAME = os.getenv("MODEL_NAME", "pretrained/base_model")