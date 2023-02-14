# #################################################################################################################### #
#       ./helpers/urban_df.py                                                                                          #
#           Utility functions for the URBANSOUND dataset.                                                              #
# #################################################################################################################### #

# OS and filesystem
from pathlib import Path

# Math
import numpy

# Sound processing
import librosa

# Misc.
import typing


# ### File paths #######################################################################################################
def get_full_path(dataset_folder: Path, fold: int, file_name: str) -> Path:
    """ Returns the full path to the audio file. """
    return dataset_folder / "audio" / f"fold{fold}" / file_name


# ### Audio loading ####################################################################################################
def load_audio(dataset_folder: Path, fold: int, file_name: str, mono: bool = True, sampling_rate: typing.Union[int, None] = 22050) -> tuple[numpy.ndarray, int]:
    """ Loads an audio file and returns the samples and the sampling rate. """
    file_path = get_full_path(dataset_folder=dataset_folder, fold=fold, file_name=file_name)
    samples, sampling_rate = librosa.load(file_path, mono=mono, sr=sampling_rate)

    return samples, sampling_rate
