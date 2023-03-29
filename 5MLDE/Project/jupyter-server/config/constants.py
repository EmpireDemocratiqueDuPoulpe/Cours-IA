# #################################################################################################################### #
#       ./config/constants.py                                                                                          #
#           Do I really need to explain this file?                                                                     #
# #################################################################################################################### #

# OS and filesystem
from pathlib import Path

# Logging
import logging


# ### Functions ########################################################################################################
def is_docker() -> bool:
    """
        Checks whenever this script is running in a Docker container.
        Reference: https://github.com/sindresorhus/is-docker/blob/main/index.js
    """
    cgroup_path = Path("/proc/self/cgroup")
    return Path("/.dockerenv").is_file() or (cgroup_path.is_file() and (cgroup_path.read_text().find("docker") > -1))


# ### Logging ##########################################################################################################
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """ Returns an initialized logger. """
    # Initialize the logger
    logger = logging.getLogger(name=name)
    logger.setLevel(level)

    # Initialize the stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)

    # Initialize the formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)

    # Return the logger
    logger.addHandler(stream_handler)
    return logger


LOG_LEVEL = logging.INFO
LOGGER = get_logger(name="model_deployment_logger", level=LOG_LEVEL)

# ### Filepaths ########################################################################################################
PARENT_FOLDER = Path("/app")
CONFIG_FOLDER = (PARENT_FOLDER / "config")
DATA_FOLDER = (PARENT_FOLDER / "data")
MODELS_FOLDER = (PARENT_FOLDER / "models")
MLFLOW_MODELS_FOLDER = Path("/mlflow")
TEMP_FOLDER = (PARENT_FOLDER / "temp")

# ### Datasets #########################################################################################################
DATASET_BASE_NAME = "Android_Malware"

# ### Model ############################################################################################################
MODEL_VERSION = "1.0.0"

# ### MLFLow ###########################################################################################################
MLFLOW_TRACKING_URI = f"http://{'mlflow' if is_docker() else 'localhost'}:5000"
MLFLOW_EXPERIMENT_NAME = "malware_prediction"
MLFLOW_MODEL_NAME = "malware_prediction_2023"
