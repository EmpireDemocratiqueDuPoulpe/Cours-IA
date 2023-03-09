import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import List
from scipy.sparse import csr_matrix
from prefect import task
from pydantic import BaseModel, validator

import constants


#### Custom types ##################################################################
class SplitDataset(BaseModel):
    x: csr_matrix
    y: np.ndarray | None = None
    dv: DictVectorizer
    
    @validator("y")
    def check_datasets_shape(cls, v, values, **kwargs) -> np.ndarray | None:
        if (v is not None) and ("x" in values) and (v.shape[0] != values["x"].shape[0]):
            raise ValueError(f"The X and Y datasets have different shapes! (x: {values['x'].shape[0]} | y: {v.shape[0]})")
        
        return v


#### Tasks - Data processing #######################################################
@task(name="load_data", tags=["load"])
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@task(name="compute_target", description="Compute the trip duration in minutes based on pickup and dropoff time.", tags=["data_processing"])
def compute_target(df: pd.DataFrame, pickup_column: str = "tpep_pickup_datetime", dropoff_column: str = "tpep_dropoff_datetime") -> pd.DataFrame:
    """
    Compute the trip duration in minutes based
    on pickup and dropoff time
    """
    df["duration"] = df[dropoff_column] - df[pickup_column]
    df["duration"] = df["duration"].dt.total_seconds() / 60
    return df


@task(name="filter_outliers", description="Remove rows corresponding to negative/zero and too high target' values from the dataset.", tags=["data_processing"])
def filter_outliers(df: pd.DataFrame, min_duration: int = 1, max_duration: int = 60) -> pd.DataFrame:
    """
    Remove rows corresponding to negative/zero
    and too high target' values from the dataset
    """
    return df[df['duration'].between(min_duration, max_duration)]


@task(name="encode_categorical_cols", description="Takes a Pandas dataframe and a list of categorical column names, and returns dataframe with the specified columns converted to categorical data type.", tags=["data_processing"])
def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    """
    Takes a Pandas dataframe and a list of categorical
    column names, and returns dataframe with
    the specified columns converted to categorical data type
    """
    if categorical_cols is None:
        categorical_cols = constants.CATEGORICAL_COLS
    df[categorical_cols] = df[categorical_cols].fillna(-1).astype('int')
    df[categorical_cols] = df[categorical_cols].astype('category')
    return df


@task(name="extract_x_y", description="Turns lists of mappings (dicts of feature names to feature values) into sparse matrices for use with scikit-learn estimators using Dictvectorizer object.", tags=["data_processing"])
def extract_x_y(df: pd.DataFrame, categorical_cols: List[str] = None, dv: DictVectorizer = None, with_target: bool = True) -> dict:
    """
    Turns lists of mappings (dicts of feature names to feature values)
    into sparse matrices for use with scikit-learn estimators
    using Dictvectorizer object.
    :return The sparce matrix, the target' values if needed and the
    dictvectorizer object.
    """
    if categorical_cols is None:
        categorical_cols = constants.CATEGORICAL_COLS
    dicts = df[categorical_cols].to_dict(orient='records')

    y = None
    if with_target:
        if dv is None:
            dv = DictVectorizer()
            dv.fit(dicts)
        y = df["duration"].values

    x = dv.transform(dicts)
    return {'x': x, 'y': y, 'dv': dv}


def process_data(path: str,  dv=None, with_target: bool = True) -> dict:
    """
    Load data from a parquet file
    Compute target (duration column) and apply threshold filters (optional)
    Turn features to sparce matrix
    :return The sparce matrix, the target' values and the
    dictvectorizer object if needed.
    """
    df = load_data(path)
    if with_target:
        df1 = compute_target(df)
        df2 = filter_outliers(df1)
        df3 = encode_categorical_cols(df2)
        return extract_x_y(df3, dv=dv)
    else:
        df1 = encode_categorical_cols(df)
        return extract_x_y(df1, dv=dv, with_target=with_target)


#### Tasks - Model training and inference ##########################################
@task(name="load_pickle", tags=["load", "serialize"])
def load_pickle(path: str):
    with open(path, 'rb') as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


@task(name="save_pickle", tags=["save", "serialize"])
def save_pickle(path: str, obj: dict):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


@task(name="train_model", description="Train and return a linear regression model.", tags=["training"])
def train_model(x_train: csr_matrix, y_train: np.ndarray) -> LinearRegression:
    """Train and return a linear regression model"""
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    return lr


@task(name="predict_duration", description="Use trained linear regression model to predict target from input data.", tags=["inference"])
def predict_duration(input_data: csr_matrix,model: LinearRegression) -> np.ndarray:
    """
    Use trained linear regression model
    to predict target from input data
    :return array of predictions
    """
    return model.predict(input_data)


@task(name="evaluate_model", description="Calculate mean squared error for two arrays.", tags=["evaluating"])
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error for two arrays"""
    return mean_squared_error(y_true, y_pred, squared=False)


def train_and_predict(x_train, y_train, x_test,  y_test) -> dict:
    """Train model, predict values and calculate error"""
    model = train_model(x_train, y_train)
    prediction = predict_duration(x_test, model)
    mse = evaluate_model(y_test, prediction)
    return {'model': model, 'mse': mse}
