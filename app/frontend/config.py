import os

IMAGES_PER_PAGE = 21
MAX_GALLERY_COLUMNS = 7
PREDICTION_BACKEND_ENDPOINT = os.getenv("PREDICTION_BACKEND_ENDPOINT", "http://localhost:8000/predict")
LOCAL_XAI_ENDPOINT = os.getenv("LOCAL_XAI_ENDPOINT", "http://localhost:8000/xai/local")
GLOBAL_XAI_ENDPOINT = os.getenv("GLOBAL_XAI_ENDPOINT", "http://localhost:8000/xai/global")