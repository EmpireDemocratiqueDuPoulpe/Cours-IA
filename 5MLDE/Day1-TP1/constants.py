from pathlib import Path

#### Paths #########################################################################
ROOT_PATH = Path("./")

TRAIN_PATH = ROOT_PATH / "data/yellow_tripdata_2021-01.parquet"
TEST_PATH = ROOT_PATH / "data/yellow_tripdata_2021-02.parquet"
INFERENCE_PATH = ROOT_PATH / "data/yellow_tripdata_2021-03.parquet"

LOCAL_STORAGE = "data/"

#### Preprocessing #################################################################
CATEGORICAL_COLS = ["PULocationID", "DOLocationID"]
