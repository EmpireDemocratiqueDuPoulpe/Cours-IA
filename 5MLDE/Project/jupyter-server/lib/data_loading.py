# #################################################################################################################### #
#       ./lib/data_loading.py                                                                                          #
#           Handles the loading of a dataset.                                                                          #
# #################################################################################################################### #

# OS and filesystem
from pathlib import Path
import sys
sys.path.append(".")

# Data
import pandas

# Local files
from config import constants


# ### Functions ########################################################################################################
def load_data(path: Path, samples: int | None = 5_000) -> pandas.DataFrame:
    """ Loads the dataset. """
    constants.LOGGER.info(f"Loading the dataset at \"{path}\"")
    data = pandas.read_csv((constants.DATA_FOLDER / "Android_Malware.csv"), index_col=0, delimiter=",", dtype={
        "Flow ID": str,
        "Source IP": str,
        "Source Port": int,
        "Destination IP": str,
        "Destination Port": int,
        "Protocol": float,
        "Timestamp": str,
        "Flow Duration": float,
        "Total Fwd Packets": float,
        "Total Backward Packets": float,
        "Total Length of Fwd Packets": float,
        "Total Length of Bwd Packets": float,
        "Fwd Packet Length Max": float,
        "Fwd Packet Length Min": float,
        "Fwd Packet Length Mean": float,
        "Fwd Packet Length Std": float,
        "Bwd Packet Length Max": float,
        "Bwd Packet Length Min": float,
        "Bwd Packet Length Mean": float,
        "Bwd Packet Length Std": float,
        "Flow Bytes/s": float,
        "Flow Packets/s": float,
        "Flow IAT Mean": float,
        "Flow IAT Std": float,
        "Flow IAT Max": float,
        "Flow IAT Min": float,
        "Fwd IAT Total": float,
        "Fwd IAT Mean": float,
        "Fwd IAT Std": float,
        "Fwd IAT Max": float,
        "Fwd IAT Min": float,
        "Bwd IAT Total": float,
        "Bwd IAT Mean": float,
        "Bwd IAT Std": float,
        "Bwd IAT Max": float,
        "Bwd IAT Min": float,
        "Fwd PSH Flags": float,
        "Bwd PSH Flags": float,
        "Fwd URG Flags": float,
        "Bwd URG Flags": float,
        "Fwd Header Length": float,
        "Bwd Header Length": float,
        "Fwd Packets/s": float,
        "Bwd Packets/s": float,
        "Min Packet Length": float,
        "Max Packet Length": float,
        "Packet Length Mean": float,
        "Packet Length Std": float,
        "Packet Length Variance": float,
        "FIN Flag Count": float,
        "SYN Flag Count": float,
        "RST Flag Count": float,
        "PSH Flag Count": float,
        "ACK Flag Count": float,
        "URG Flag Count": float,
        "CWE Flag Count": "object",  # Change later
        "ECE Flag Count": float,
        "Down/Up Ratio": "object",  # Change later
        "Average Packet Size": float,
        "Avg Fwd Segment Size": float,
        "Avg Bwd Segment Size": float,
        "Fwd Header Length.1": float,
        "Fwd Avg Bytes/Bulk": "object",  # Change later
        "Fwd Avg Packets/Bulk": float,
        "Fwd Avg Bulk Rate": float,
        "Bwd Avg Bytes/Bulk": float,
        "Bwd Avg Packets/Bulk": float,
        "Bwd Avg Bulk Rate": float,
        "Subflow Fwd Packets": float,
        "Subflow Fwd Bytes": float,
        "Subflow Bwd Packets": float,
        "Subflow Bwd Bytes": float,
        "Init_Win_bytes_forward": float,
        "Init_Win_bytes_backward": float,
        "act_data_pkt_fwd": float,
        "min_seg_size_forward": float,
        "Active Mean": float,
        "Active Std": float,
        "Active Max": float,
        "Active Min": float,
        "Idle Mean": float,
        "Idle Std": float,
        "Idle Max": float,
        "Idle Min": float,
        "Label": str
    })

    # Sanitize the dataset
    data = sanitize_dataset(data)

    if samples is not None:
        data = data.sample(n=samples)

    return data


def sanitize_dataset(df: pandas.DataFrame) -> pandas.DataFrame:
    """ Clears out the dataset. """
    df_sanitized = df.copy()
    df_sanitized.columns = df_sanitized.columns.str.strip()

    # Drop duplicates and missing values
    df_sanitized = df_sanitized.drop_duplicates(keep="first", inplace=False)
    df_sanitized = df_sanitized.dropna(inplace=False)

    # Clean the "CWE Flag Count" column
    df_sanitized.drop(df_sanitized.loc[df_sanitized["CWE Flag Count"] == "SCAREWARE"].index, inplace=True)
    df_sanitized["CWE Flag Count"] = df_sanitized["CWE Flag Count"].astype(int)

    # Clean the "Fwd Avg Bytes/Bulk" column
    df_sanitized.drop(df_sanitized.loc[df_sanitized["Fwd Avg Bytes/Bulk"] == "BENIGN"].index, inplace=True)
    df_sanitized["Fwd Avg Bytes/Bulk"] = df_sanitized["Fwd Avg Bytes/Bulk"].astype(float)

    # Clean the "Down/Up Ratio" column
    df_sanitized.drop(df_sanitized.loc[df_sanitized["Down/Up Ratio"] == "BENIGN"].index, inplace=True)
    df_sanitized["Down/Up Ratio"] = df_sanitized["Down/Up Ratio"].astype(float)

    return df_sanitized
