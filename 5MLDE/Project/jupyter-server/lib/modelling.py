# #################################################################################################################### #
#       ./lib/modelling.py                                                                                             #
#           Functions used to train the model and fetch prediction.                                                    #
# #################################################################################################################### #

# OS and filesystem
from pathlib import Path
import joblib

# Math
import numpy

# Data
import pandas

# Model processing
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

# Logging
import logging

# Misc.
from prefect import task

# Local files
from .data_preprocessing import get_preprocessor


# ### Training #########################################################################################################
@task(name="train_model", tags=["model", "train"])
def train_model(x_train: numpy.ndarray, y_train: numpy.ndarray) -> Pipeline:
    """ Fits the model on the train set. """
    preprocessor = get_preprocessor()
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("k_neighbors_classifier", KNeighborsClassifier())
    ])

    pipeline.fit(x_train, y_train)
    return pipeline


# ### Predicting #######################################################################################################
@task(name="get_predictions", tags=["model", "prediction"])
def get_predictions(pipeline: Pipeline, x_test: numpy.ndarray, logger: logging.Logger) -> pandas.Series:
    """ Uses a trained model to build a prediction array. """
    logger.info(msg="Making predictions...")

    predictions = pipeline.predict(x_test)
    return predictions


# ### Inference ########################################################################################################
@task(name="run_inference", tags=["model", "prediction"])
def run_inference(pipeline: Pipeline, payload: dict, logger: logging.Logger) -> str:
    """ Predicts if a phone is infected by a malware. """
    logger.info(msg="Running inference on payload...")
    return pipeline.predict(pandas.DataFrame([payload]))[0]


# ### Metrics ##########################################################################################################
@task(name="compute_accuracy", tags=["model", "metrics computing"])
def compute_accuracy(pipeline: Pipeline, x: numpy.ndarray, y: numpy.ndarray, logger: logging.Logger) -> float:
    """ Compute the model accuracy. """
    logger.info(msg="Computing accuracy...")
    return pipeline.score(x, y)


# ### Saving/loading ###################################################################################################
@task(name="save_pipeline", tags=["model", "disk interaction"])
def save_pipeline(pipeline: Pipeline, path: Path, logger: logging.Logger) -> None:
    """ Saves the pipeline to the disk. """
    logger.info(msg=f"Saving pipeline to \"{path}\"...")
    joblib.dump(value=pipeline, filename=path)


@task(name="load_pipeline", tags=["model", "disk interaction"])
def load_pipeline(path: Path, logger: logging.Logger) -> Pipeline:
    """ Loads the pipeline from the disk. """
    logger.info(msg=f"Loading pipeline from \"{path}\"...")
    return joblib.load(filename=path)
