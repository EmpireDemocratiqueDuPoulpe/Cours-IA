# #################################################################################################################### #
#       ./lib/data_preprocessing.py                                                                                    #
#           Preprocess the data before the training. It assumes that the dataset is already sanitized.                 #
# #################################################################################################################### #

# Data
import pandas

# Model processing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Misc.
import typing
from prefect import task


# ### Functions ########################################################################################################
@task(name="prepare_data", tags=["data", "preprocessing"])
def prepare_data(df: pandas.DataFrame) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
    """ Prepare the data for the training. Returns (x_train, x_test, y_train, y_test). """
    df = drop_unused_columns(df)

    # Split features
    data_x = df.drop(["Label"], axis=1)
    data_y = df["Label"]

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.20)
    return x_train, x_test, y_train, y_test


# ### Column cleaning ##################################################################################################
def drop_unused_columns(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.drop(["Flow ID"], axis=1, inplace=False)


# ### Pipeline #########################################################################################################
def get_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_features = ["Source Port", "Destination Port", "Protocol", "Flow Duration", "Total Fwd Packets",
                        "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets",
                        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
                        "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
                        "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
                        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total",
                        "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
                        "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
                        "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
                        "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
                        "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
                        "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
                        "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
                        "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
                        "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
                        "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
                        "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
                        "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]
    categorical_features = ["Source IP", "Destination IP", "Timestamp"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor
