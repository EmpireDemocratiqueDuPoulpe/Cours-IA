import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

from lib.modelling import run_inference
from app_config import (APP_TITLE, APP_DESCRIPTION, APP_VERSION, MLFLOW_TRACKING_URI, REGISTERED_MODEL_URI)


app = FastAPI(title=APP_TITLE,
              description=APP_DESCRIPTION,
              version=APP_VERSION)


class InputData(BaseModel):
    PULocationID: int
    DOLocationID: int
    passenger_count: int


class PredictionOut(BaseModel):
    trip_duration_prediction: float


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
pipeline = mlflow.pyfunc.load_model(model_uri=REGISTERED_MODEL_URI)


@app.get("/")
def home():
    return {"health_check": "OK",
            "model_version": pipeline._model_meta.run_id}


@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: InputData):
    trip_duration_prediction = run_inference(payload.dict(), pipeline)
    return {"trip_duration_prediction": trip_duration_prediction}
