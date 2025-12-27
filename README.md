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
    - [Other Modules](#other-modules)
        - [DVC PIPELINES](#dvc-pipelines)
        - [INFRASTRUCTURE](#infrastructure)

## Project Structure

```plaintext
CLASSIFICATION_XAI_DEMO/
├── app/
│   ├── backend/
│   │   ├── api/
│   │   │   ├── metrics.py
│   │   │   ├── predict.py
│   │   ├── config.py
│   │   ├── main.py
│   ├── frontend/
│   │   ├── components/
│   │   │   ├── gallery.py
│   │   │   ├── selected_view.py
│   │   │   ├── utils.py
│   │   ├── config.py
│   │   ├── home.py
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
│   │   │   ├── ...
│   │   │   ├── graphs.py
│   │   ├── data/
│   │   │   ├── ...
│   │   │   ├── big_cats.py
│   │   ├── models/
│   │   │   ├── ...
│   │   │   ├── alexnet.py
│   │   ├── pipelines/
│   │   │   ├── ...
│   │   │   ├── pipeline.py
│   ├── config.py
│   ├── big_cats_evaluate.py
│   ├── big_cats_train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How to use

### Setup

For setting up the repo after cloning, install the dependecies and prepare the data.

```
pip install -r requirements.txt
```

Download the data manually.
- [Link to 10 Big Cats dataset](https://www.kaggle.com/datasets/gpiosenka/cats-in-the-wild-image-classification)

### Deployment

The deployment of the demo involves training a model on the dataset and launching the application, both the frontend and the backend.

#### Manual deployment

##### Training and evaluating the model

The source module includes a training file that allows to train the model directly. If no model is trained, simply run:

```
python -m src.big_cats_train
```

The other source script included is the evaluation script. It allows to evaluate the model on the test set and register the results.

Additionally, use the `--model-name` flag to specify the name of the model to be evaluated. If no model is specified, the script will look for the latest model in the `models` directory.

```
python -m src.big_cats_evaluate --model-name <model_name>
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
python -m streamlit run app/frontend/home.py
```

#### Docker deployment

Inside the `infrastructure/docker` folder there are the required components to ease the deployment process using docker.

The Docker deployment uses the **pretrained model** (`models/pretrained/base_model`) by default, so no training is required.

Optionally, one can build the docker images manually:

```
docker build -f infrastructure/docker/Dockerfile.backend -t mlops-api:latest .
docker build -f infrastructure/docker/Dockerfile.frontend -t mlops-web:latest .
```

Running the compose will launch both the backend and the frontend, and will also build the required images if they haven't been built before. The compose also deploys prometheus and grafana containers, configured through the `infrastructure/monitoring` module, for monitoring purposes.

```
docker compose -f infrastructure/docker/docker-compose.yaml --env-file infrastructure/.env up
```

A .env file is used in the `infrastructure` folder. Any external port changes can be done through this file and the docker-compose. Understand, the internal ports are hardcoded and changing them in one place might imply changes in other documents.

> **Note**: The backend uses the pretrained model (`models/pretrained/base_model`) by default. To use a different model, set the `MODEL_NAME` variable in the backend configuration, the docker-compose.yaml or .env file.
>
> Remember, if you want to train a new model locally before deployment, you can use the training script. [See the training section](#training-and-evaluating-the-model).

## Modules - Specs

The project is divided into various individual modules. Inside the modules there are several low_level and mid-level sepparated configuration files, in order to allow for different levels of changes that control the behaviour of each module.

### APP

This is the module that contains the application, both the frontend and the backend.

#### Backend

The backend is a Flask API that allows for requesting predictions and other features. It is divided into the actual endpoints, stored in the `api/` folder, and the `main.py` file, which registers the endpoints and launches the backend itself.

##### Configuration

The configuration of the backend can be found at `app/backend/`.

- ``BACKEND_PORT``: The default port for backend to be launched on.
- ``MODEL_NAME``: The name of the model to be used for predictions. If not specified, the latest model will be used.

#### Frontend

The frontend is a simple streamlit app, which presents different samples of the data to the user and allows to operate over them by requesting to the model of the backend through the API.

The ``home.py`` file loads the main page based on the different utilities and components found at ``app/frontend/components``.

##### Configuration

The configuration of the frontend can be found at `app/frontend/`.

- ``IMAGES_PER_PAGE``: Maximum number of images per batch on display.
- ``MAX_GALLERY_COLUMNS``: Maximum number of columns allowed for the display of the images. 
- ``PREDICTION_BACKEND_ENDPOINT``: The URL of the API endpoint for making predictions.

### SRC

This module contains the source code. It is divided between the actionable scripts found in  `src/` and the code core, and expandable framework on which to build any new capabilities for the project (found at `scr/core/`)

The `core` is divided into modules that encapsulate different necessities, such as `data`, `models` or `pipelines`. In the end, the goal is to be able to create an easy to build modular stream from the executable scripts. Each of the sub-modules in the `core` have their specific configuration files, which are not intended to be modiffied by the user, but rather were created for developing porpuses. Either way, their configurations will be expanded upon after the user configuration file.

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

- ``BIG_CATS_PIPELINE_CONFIG``: The dictionary used to configure the pipeline in the executables. Set as a configuration for reusability.
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

##### Data configuration

- ``DATA_DIR``: The directory of the root path from which to load the data.

- ``BIG_CATS_DIR``: Name of the folder inside the data directory on which the data from the Pediatric Pneumonia Chest X-ray dataset is saved at.
- ``BIG_CATS_TRANSFORMS``: A list containing the transformations for the dataset images.

##### Pipelines configuration

- ``MODELS_SAVE_DIR``: The directory from the root on which to save the models.
- ``METRICS_DIR``: The directory from the root on which the metric logs are saved. Used by visualization solutions like tensorboard.

##### Analysis configuration

- ``ANALYSIS_DIR``: The directory from the root on which the analysis results are saved.
- ``GRAPH_DIR``: The directory from the root on which the analysis graphs are saved.


#### INFRASTRUCTURE

This module contains the infrastructure code for the deployment of the demo at multiple levels. The submodules contained in it support each other for the deployments.

Current infrastructure:
- `Docker`: COntains the neccesary files for the dockerization of the application, both frontend and backend, as well as for simultaneos deployment of the app and and the monitorization with Prometheus and Grafana.
- `Monitoring`: Contains the neccesary files for the deployment of the monitoring modules, both Prometheus and Grafana.
- `.env`: A configuration file used for configuration regarding all the infrastructure. Mainly for setting the external exposed ports, both when deploying locally with docker, as well as the external ports of the services on GKE.