import logging

import pandas as pd


def get_logger(logging_level=logging.INFO, logger_name: str = "app_logger"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# LOGGING
LOGGER_LEVEL = "INFO"
logger = get_logger(logging_level=getattr(logging, LOGGER_LEVEL))


# MODELS
CATEGORICAL_VARS = ['PULocationID', 'DOLocationID', 'passenger_count']

# MLFLOW
MLFLOW_TRACKING_URI = "http://host.docker.internal:5001"
#MLFLOW_TRACKING_URI = "http://mlflow:5001"

STAGE = "Production"
REGISTERED_MODEL_NAME = "trip_duration_model"
REGISTERED_MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}/{STAGE}"

# MISC
APP_TITLE = "TripDurationPredictionApp"
APP_DESCRIPTION = ("A simple API to predict trip duration in minutes "
                   "for NYC yellow taxi trips, given a pickup, a dropoff location "
                   "and a passenger count. Using a model pulled from MLFlow.")
APP_VERSION = "0.0.1"
# silence pandas `SettingWithCopyWarning` warnings
pd.options.mode.chained_assignment = None  # default='warn'
