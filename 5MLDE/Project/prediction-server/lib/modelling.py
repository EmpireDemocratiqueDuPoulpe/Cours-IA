# #################################################################################################################### #
#       ./lib/modelling.py                                                                                             #
#           Functions used to train the model and fetch prediction.                                                    #
# #################################################################################################################### #

# OS and filesystem
from pathlib import Path
import joblib

# Data
import pandas

# Model processing
from sklearn.pipeline import Pipeline

# Logging
import logging


# ### Inference ########################################################################################################
def run_inference(pipeline: Pipeline, payload: dict, logger: logging.Logger) -> str:
    """ Predicts if a phone is infected by a malware. """
    logger.info(msg="Running inference on payload...")
    return pipeline.predict(pandas.DataFrame([payload]))[0]


# ### Saving/loading ###################################################################################################
def load_pipeline(path: Path, logger: logging.Logger) -> Pipeline:
    """ Loads the pipeline from the disk. """
    logger.info(msg=f"Loading pipeline from \"{path}\"...")
    return joblib.load(filename=path)
