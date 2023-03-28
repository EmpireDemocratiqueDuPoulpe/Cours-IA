# #################################################################################################################### #
#       ./web_service/internal_config.py                                                                               #
#           Internal config of the webservice. Some constants may be replicated from ROOT/config/*                     #
# #################################################################################################################### #

# OS and filesystem
from pathlib import Path

# Logging
import logging


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
LOGGER = get_logger(name="web_service_logger", level=LOG_LEVEL)

# ### Filepaths ########################################################################################################
PARENT_FOLDER = Path.cwd()
MODELS_FOLDER = (PARENT_FOLDER / "local_models").resolve()

# ### Model ############################################################################################################
MODEL_VERSION = "1.0.0"
LAST_MODEL_PATH = (MODELS_FOLDER / f"pipeline__v{MODEL_VERSION}.joblib")

# ### API stuff ########################################################################################################
APP_TITLE = "Is your phone infected?"
APP_DESCRIPTION = (
    "You have downloaded RAM and now your phone shows weird cyrillic symbols everywhere? Your Facebook™ account is "
    "constantly sending posts about winning a free trip to Barcelona? Maybe you even tried to install this .apk file "
    "named \"Grand.Theft.Auto.VI.Mobile.4K.Not.A.Scam\"? Says no more, we will tell you if your phone has been invaded "
    "by an evil virus."
    " "
    " "
    "¹'³²﹪ ᵒᶠ ᵃᶜᶜᵘʳᵃᶜʸ"
)
APP_VERSION = "1.0.0"
