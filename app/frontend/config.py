import os

IMAGES_PER_PAGE = 21
MAX_GALLERY_COLUMNS = 7
PREDICTION_BACKEND_ENDPOINT = os.getenv("PREDICTION_BACKEND_ENDPOINT", "http://localhost:8000/predict")
LOCAL_XAI_ENDPOINT = os.getenv("LOCAL_XAI_ENDPOINT", "http://localhost:8000/xai/local")
GLOBAL_XAI_ENDPOINT = os.getenv("GLOBAL_XAI_ENDPOINT", "http://localhost:8000/xai/global")
DATASET_SELECTION_ENDPOINT = os.getenv("DATASET_SELECTION_ENDPOINT", "http://localhost:8000/dataset/select")
DATASET_CURRENT_ENDPOINT = os.getenv("DATASET_CURRENT_ENDPOINT", "http://localhost:8000/dataset/current")

DATASET_METADATA = {
    "stl10": {
        "display_name": "STL10",
        "description": "10 classes of common objects: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck",
        "num_classes": 10,
        "icon": "✈️"
    },
    "big_cats": {
        "display_name": "Big Cats",
        "description": "10 species of wild cats: African Leopard, Caracal, Cheetah, Clouded Leopard, Jaguar, Lions, Ocelot, Puma, Snow Leopard, Tiger",
        "num_classes": 10,
        "icon": "🐆"
    }
}