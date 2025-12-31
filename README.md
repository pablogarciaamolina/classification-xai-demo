# Classification XAI Demo
Application demo for prediction and XAI assesment using the 10 Big Cats dataset.

## Table of Contents

1. [Project Structure](#project-structure)
2. [How To Use](#how-to-use)
    - [Setup](#setup)
    - [Deployment](#deployment)
        - [Manual deployment](#manual-deployment)
            - [Training and evaluating the model](#training-and-evaluating-the-model)
            - [Visualizing results](#visualizing-results)
            - [Launching the app](#launching-the-app)
        - [Docker deployment](#docker-deployment)
3. [Modules - Specs](#modules---specs)
    - [APP](#app)
        - [Backend](#backend)
        - [Frontend](#frontend)
    - [SRC](#src)
        - [Core](#core)
            - [Analysis](#analysis)
            - [Data](#data)
            - [Models](#models)
            - [Pipelines](#pipelines)

## Project Structure

```plaintext
CLASSIFICATION_XAI_DEMO/
├── app/
│   ├── backend/
│   │   ├── api/
│   │   │   ├── metrics.py
│   │   │   ├── predict.py
│   │   │   ├── xai.py
│   │   ├── utils/
│   │   │   ├── xai_service.py
│   │   ├── config.py
│   │   ├── main.py
│   ├── frontend/
│   │   ├── components/
│   │   │   ├── gallery.py
│   │   │   ├── saliency_viz.py
│   │   │   ├── selected_view.py
│   │   │   ├── utils.py
│   │   ├── pages/
│   │   │   ├── home.py
│   │   │   ├── global_xai.py
│   │   │   ├── local_xai.py
│   │   ├── config.py
│   │   ├── main.py
├── data/
│   ├── ...
├── infrastructure/
│   ├── docker/
│   │   ├── docker-compose.yaml
│   │   ├── Dockerfile.backend
│   │   ├── Dockerfile.frontend
│   ├── monitoring/
│   │   ├── grafana-provisioning/
│   │   │   ├── datasources/
│   │   │   │   ├── prometheus.yaml
│   │   ├── prometheus.yaml
│   ├── .env
├── models/
│   ├── ...
├── src/
│   ├── core/
│   │   ├── analysis/
│   │   │   ├── metrics/
│   │   │   │   ├── ...
│   │   │   │   ├── graphs.py
│   │   │   ├── xai/
│   │   │   │   ├── evaluations/
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── average_sensitivity.py
│   │   │   │   │   ├── road.py
│   │   │   │   ├── methods/
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── cam.py
│   │   │   │   │   ├── gradient_ascent.py
│   │   │   │   │   ├── integrated_gradients.py
│   │   │   ├── ...
│   │   ├── data/
│   │   │   ├── ...
│   │   │   ├── big_cats.py
│   │   │   ├── garbage.py
│   │   │   ├── pediatric_pneumonia.py
│   │   │   ├── stl10.py
│   │   ├── models/
│   │   │   ├── ...
│   │   │   ├── alexnet.py
│   │   │   ├── resnet.py
│   │   ├── pipelines/
│   │   │   ├── ...
│   │   │   ├── pipeline.py
│   ├── config.py
│   ├── big_cats_evaluate.py
│   ├── big_cats_train.py
├── .gitignore
├── README.md
├── requirements.txt
└── xai_use_example.ipynb
```

## How to use

### Setup

For setting up the repo after cloning, install the dependecies and prepare the data.

```
pip install -r requirements.txt
```

Download the data manually.
- STL dataset is already supported natively by torchvision.
- [Link to 10 Big Cats dataset](https://www.kaggle.com/datasets/gpiosenka/cats-in-the-wild-image-classification)
- [Link to Pediatric Pneumonia dataset](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)

### Deployment

The deployment of the demo involves training a model on the dataset and launching the application, both the frontend and the backend.

#### Manual deployment

##### Training and evaluating the model

The source module includes a training file that allows to train the model directly, using flags to select the data (big_cats, stl10, pediatric_pneumonia) and the architecture (alexnet, resnet). If no model is trained, simply run:

```
python -m src.train --data <dataset name> --model <architectur name>
```

The other source script included is the evaluation script. It allows to evaluate the model on the test set and register the results. It also support flagging for dataset and saved model selection. If no model is specified, the script will look for the latest model in the `models` directory.

```
python -m src.evaluate --data <dataset name> --model-name <model name>
```

##### Visualizing results

The training is integrated with MLflow. Launch a local server to visualize the experiments with:

```
mlflow ui
```

The visualizations of the training graphs are also avaliable through tensorboard:

```
tensorboard --logdir=runs
```

##### Launching the app

The frontend and the backend of the app are concieved to be deployed independently from each other. The commands are the following:

```
python -m app.backend.main
```
```
python -m streamlit run app/frontend/main.py
```

## Modules - Specs

The project is divided into various individual modules. Inside the modules there are several low_level and mid-level sepparated configuration files, in order to allow for different levels of changes that control the behaviour of each module.

### APP

This is the module that contains the application, both the frontend and the backend.

#### Backend

The backend is a Flask API that allows for requesting predictions and other features. It is divided into the actual endpoints, stored in the `api/` folder, a set of utilities like the specific XAI pipeline (`utils\`), and the `main.py` file, which registers the endpoints and launches the backend itself.

##### Configuration

The configuration of the backend can be found at `app/backend/`.

- ``BACKEND_PORT``: The default port for backend to be launched on.
- ``ROAD_PERCENTILES``: The percentiles to be used for ROAD.
- ``AVERAGE_SENSITIVITY_SAMPLES``: The number of samples to be used for the average sensitivity calculation.

- ``DATASET_CONFIGS``: The configuration of the datasets supported by the application.
    - ``stl10`` {
        - **model_name**: The name of the model to be used for predictions.
        - **num_classes**: Number of classes in the dataset.
        - **in_channels**: Number of channels in the dataset.
        - **image_size**: Size of the images in the dataset.
        - **gradcam_target_layer**: The target layer for Grad-CAM.
        - **small_inputs**: Whether the inputs are small or not. Used for STL10.
        - **classes**: List of classes in the dataset.
        - **pipeline_config**: Configuration for the XAI pipeline.
    },
    - ``big_cats``: 10 Big Cats dataset
        ...(same as above)...

#### Frontend

The frontend is a simple streamlit app, which presents different samples of the data to the user and allows to operate over them by requesting to the model of the backend through the API.

The ``main.py`` file loads the different pages from ``app/frontend/pages`` based on the different utilities and components found at ``app/frontend/components``.

##### Configuration

The configuration of the frontend can be found at `app/frontend/`.

- ``IMAGES_PER_PAGE``: Maximum number of images per batch on display.
- ``MAX_GALLERY_COLUMNS``: Maximum number of columns allowed for the display of the images. 
- ``PREDICTION_BACKEND_ENDPOINT``: The URL of the API endpoint for making predictions.
- ``LOCAL_XAI_ENDPOINT``: The URL of the API endpoint for making local XAI requests.
- ``GLOBAL_XAI_ENDPOINT``: The URL of the API endpoint for making global XAI requests.
- ``DATASET_SELECTION_ENDPOINT``: The URL of the API endpoint for selecting a dataset.
- ``DATASET_CURRENT_ENDPOINT``: The URL of the API endpoint for getting the current dataset.

- ``DATASET_METADATA``: The metadata of the datasets supported by the application.
    - ``stl10``: 
        - **display_name**: The name of the dataset.
        - **description**: A description of the dataset.
        - **num_classes**: The number of classes in the dataset.
        - **icon**: An icon representing the dataset.
    - ``big_cats``: 
        ...(same as above)...

### SRC

This module contains the source code. It is divided between the actionable scripts found in  `src/` and the code **core**, and expandable framework on which to build any new capabilities for the project (found at `scr/core/`)

The `core` is divided into modules that encapsulate different necessities, such as `data`, `models`, `pipelines` and `analysis`. In the end, the goal is to be able to create an easy to build modular stream from the executable scripts. Each of the sub-modules in the `core` have their specific configuration files, which are not intended to be modiffied by the user, but rather were created for developing porpuses. Either way, their configurations will be expanded upon after the user configuration file.

#### Main configuration

This configuration can be found at `src/`. In general, the variables that are NOT in capital letters are save to change.

##### User parameters
Paramenters that the training based on.

- ``device``: The device to use for the operations.

- ``num_classes``: The number of classes for the model to predict.
- ``batch_size``: The batch size for training

- ``lr``: The base learning rate for training.
- ``epochs``: The number of epoch of training.

##### Developing configuration

This is the configuration for the executables based on the parameters. ⚠️ Wrongly changing this configuration directly, instead of doing so through the parameters, could raise errors.

- ``NUM_CLASSES``: The number of classes to predict.
- ``BATCH_SIZE``: The batch size for the data.

- ``<ANY DATASET>_PIPELINE_CONFIG``: The dictionary used to configure the pipeline in the executables. Set as a configuration for reusability.
    - **device**: The operational device.
    - **epochs**: Number of training epochs.
    - **loss_class**: Loss function class for training the model.
    - **optimizer_class**: Optimizer class for updating the model paramenters.
    - ``optimizer_kwargs``: Any parameters for the selected optimizer.
        - **lr**:  The base learning rate.
        - **weight_decay**: The weight decay of the optimizer *(If applicable)*.
    - **scheduler_class**: Scheduler class for updating the learning rate.
    - ``scheduler_kwargs``: Any parameters for the selected scheduler.
        - **eta_min**: Minimum learning rate *(If applicable)*.
        - **T_max**: Maximum number of iterations *(If applicable)*.

#### Core

⚠️ The following are lowest-level configuration files. The modification of some of the variables may drastically affect the behaviour and reproducibility of the code. Also, modifying this might change the information or steps defined on this document.

##### Analysis 

This is the main module of this project, as it contains not only the metrics methods, but all the XAI implementations. The explanation methods implemented are coded as individual classes with the neccesary capabilities to compute the explanationand present the results individually.

###### Avaliable analysis methods

- ``Metrics``
    - Confusion matrix
    - Classification report
- ``XAI``
    - ``Methods``
        - Grad-CAM
        - Gradient Ascent
        - Integrated Gradients
    - ``Evaluations``
        - Average Sensitivity
        - ROAD

###### Analysis configuration

- ``ANALYSIS_DIR``: The directory from the root on which the analysis results are saved.
- ``GRAPHS_DIR``: The directory from the root on which the analysis graphs are saved.
- ``XAI_DIR``: The directory from the root on which the XAI results are saved.

##### Data

This module cxontains all the files needed to load the datasets. It defines different classes for each of them, as well as methods for quickly loading the data.

###### Data configuration

- ``DATA_DIR``: The directory of the root path from which to load the data.

- ``<ANY DATASET>_DIR``: Name of the folder inside the data directory on which the data from the dataset is saved at.
- ``<ANY DATASET>_IMAGE_SIZE``: The size for the images in the dataset.
- ``<ANY DATASET>_TRANSFORMS``: A list containing the transformations for the dataset images when testing or displaying.
- ``<ANY DATASET>_TRAINING_TRANSFORMS``: A list containing the transformations for the dataset images when training.

##### Models

This module contains the models architechtures.

###### Avaliable models

- **AlexNet**
- **ResNet**
    - ResNet18
    - Resnet34

##### Pipelines

This module contains the pipelines that allow to seamlessly merge data and model, controlling the training and evaluation flows.

###### Avaliable pipelines

- **ClassificationPipeline**

###### Pipelines configuration

- ``MODELS_SAVE_DIR``: The directory from the root on which to save the models.
- ``METRICS_DIR``: The directory from the root on which the metric logs are saved. Used by visualization solutions like tensorboard.