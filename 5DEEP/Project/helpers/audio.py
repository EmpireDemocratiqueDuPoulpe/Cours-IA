# #################################################################################################################### #
#       ./helpers/audio.py                                                                                             #
#           Audio-related functions.                                                                                   #
# #################################################################################################################### #

# OS and filesystem
from pathlib import Path

# Math
import numpy

# Data
import pandas

# Sound processing
import librosa
import librosa.display

# Local files
from . import urban_df


# ### Samples ##########################################################################################################
def fix_samples_length(samples: numpy.ndarray, sampling_rate: int, duration_sec: int) -> numpy.ndarray:
    """ Truncate or pad a sample array to a fixed duration (in seconds). """
    required_samples = duration_sec * sampling_rate
    samples_count = len(samples)

    if samples_count < required_samples:  # Pad the audio
        return librosa.util.pad_center(data=samples, size=required_samples, mode="constant")
    elif samples_count > required_samples:  # Truncate the audio
        return librosa.util.fix_length(data=samples, size=required_samples, mode="constant")
    else:  # No operation needed
        return samples


# ### Visualizations ###################################################################################################
def make_mel_spectrogram(dataset_folder: Path, row: pandas.Series, n_fft: int = 2048, hop_length: int = None, n_mels: int = 128, duration_sec: int = None) -> numpy.ndarray:
    """ Build the Mel spectrogram of an audio file. """
    if hop_length is None:
        hop_length = (n_fft // 4)

    samples, sampling_rate = urban_df.load_audio(dataset_folder=dataset_folder, fold=row["fold"], file_name=row["slice_file_name"])

    if duration_sec:  # Pad/truncate the samples
        samples = fix_samples_length(samples=samples, sampling_rate=sampling_rate, duration_sec=duration_sec)

    # Mel spectrogram
    samples_normalized = librosa.util.normalize(samples)
    mel = librosa.feature.melspectrogram(y=samples_normalized, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.amplitude_to_db(numpy.abs(mel))
    mel_normalized = librosa.util.normalize(mel_db)

    return mel_normalized