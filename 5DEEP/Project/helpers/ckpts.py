# #################################################################################################################### #
#       ./helpers/ckpts.py                                                                                             #
#           Checkpoint-related functions.                                                                              #
# #################################################################################################################### #

# OS and filesystem
from pathlib import Path

# Data
import re

# Model processing
from tensorflow import keras


# ### Checkpoint loading ###############################################################################################
def load_best_model(models_folder: Path, base_name: str, loss_key: str = "vloss-") -> keras.Model:
    """ Loads the best model from the checkpoint files based on the validation loss value. """
    # Get all the checkpoints
    ckpt_files = [file for file in models_folder.glob(f"{base_name}*.hdf5") if file.is_file()]
    lowest_loss = None
    best_ckpt = ckpt_files[0]

    # Find the checkpoint with the lower validation loss
    for file in ckpt_files:
        val_loss = re.search(pattern=fr"{loss_key}[-+]?(?P<loss_value>\d*\.*\d+)", string=file.name)

        if val_loss:
            val_loss = float(val_loss.group("loss_value"))

            if (lowest_loss is None) or (val_loss <= lowest_loss):
                lowest_loss = val_loss
                best_ckpt = file

    # Load the best model
    return keras.models.load_model(filepath=best_ckpt, compile=True)