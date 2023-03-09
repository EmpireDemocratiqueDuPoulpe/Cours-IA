from pathlib import Path
import os
import numpy as np
import urllib.request
from prefect import flow
from prefect.task_runners import SequentialTaskRunner

import constants
import orchestration_02_first_flow as tasks


#### Data download #################################################################
def download_data() -> None:
    urllib.request.urlretrieve("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-01.parquet", constants.TRAIN_PATH)
    urllib.request.urlretrieve("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-02.parquet", constants.TEST_PATH)
    urllib.request.urlretrieve("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-03.parquet", constants.INFERENCE_PATH)


#### Model training ################################################################
@flow(name="complete_ml", description="Load data and prepare sparse matrix (using dictvectorizer) for model training. Train model, make predictions and calculate error. Save model and dictvectorizer to a folder in pickle format.", task_runner=SequentialTaskRunner)
def complete_ml(train_path: Path, test_path: Path, save_model: bool = True, save_dv: bool = True, local_storage: str = constants.LOCAL_STORAGE) -> None:
    """
    Load data and prepare sparse matrix (using dictvectorizer) for model training
    Train model, make predictions and calculate error
    Save model and dictvectorizer to a folder in pickle format
    :return none
    """
    if not os.path.exists(local_storage):
        os.makedirs(local_storage)

    train_data = tasks.process_data(train_path)
    test_data = tasks.process_data(test_path, dv=train_data["dv"])
    model_obj = tasks.train_and_predict(train_data["x"], train_data["y"], test_data['x'], test_data['y'])
    if save_model:
        tasks.save_pickle(f"{local_storage}/model.pickle", model_obj)
    if save_dv:
        tasks.save_pickle(f"{local_storage}/dv.pickle", train_data["dv"])


#### Inference #####################################################################
@flow(name="batch_inference", description="Load model and dictvectorizer from folder. Transforms input data with dictvectorizer. Predict values using loaded model.", task_runner=SequentialTaskRunner)
def batch_inference(input_path, dv=None, model=None, local_storage=constants.LOCAL_STORAGE) -> np.ndarray:
    """
    Load model and dictvectorizer from folder
    Transforms input data with dictvectorizer
    Predict values using loaded model
    :return array of predictions
    """
    if not dv:
        dv = tasks.load_pickle(f"{local_storage}/dv.pickle")
    data = tasks.process_data(input_path, dv, with_target=False)
    if not model:
        model = tasks.load_pickle(f"{local_storage}/model.pickle")["model"]
    return tasks.predict_duration(data["x"], model)


#### Main ##########################################################################
if __name__ == "__main__":
    download_data()
    complete_ml(constants.TRAIN_PATH, constants.TEST_PATH)
    inference = batch_inference(constants.INFERENCE_PATH)
