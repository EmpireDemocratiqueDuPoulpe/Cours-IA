# Model processing
import mlflow

# Local files
from config import constants
from lib.data_loading import load_data
from lib.data_preprocessing import prepare_data
from lib.modelling import train_model, get_predictions, compute_accuracy


if __name__ == "__main__":
    # Initialize MLFlow
    mlflow.set_tracking_uri(constants.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(constants.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        constants.LOGGER.info(msg=f"MLFlow - Run ID: {run.info.run_id}")
        constants.LOGGER.info(msg=f"    > Artifact URI: {mlflow.get_artifact_uri()}")
        constants.LOGGER.info(msg=f"    > Registry URI: {mlflow.get_registry_uri()}")

        # Get the data
        data = load_data(path=(constants.DATA_FOLDER / "Android_Malware_cleaned.csv"), samples=5_000)
        x_train, x_test, y_train, y_test = prepare_data(df=data)

        # Train and predict
        pipeline = train_model(x_train=x_train, y_train=y_train)
        predictions = get_predictions(pipeline=pipeline, x_test=x_test, y_test=y_test, logger=constants.LOGGER)

        # Compute metrics
        train_accuracy = compute_accuracy(x=x_train, y=y_train, logger=constants.LOGGER)
        test_accuracy = compute_accuracy(x=x_test, y=y_test, logger=constants.LOGGER)

        constants.LOGGER.info(msg=f"Model accuracy (train): {(train_accuracy * 100):.6f}%")
        constants.LOGGER.info(msg=f"Model accuracy (test): {(test_accuracy * 100):.6f}%")

        # Register the model to the MLFlow registry
        constants.LOGGER.info(msg=f"Registering the model to MLFLow with the name \"{constants.MLFLOW_MODEL_NAME}\"...")
        mlflow.log_params({
            "model_type": pipeline.steps[1][1].__class__.__name__,
            "preprocessing_type": pipeline.steps[0][1].__class__.__name__
        })
        mlflow.log_metric(key="train_accuracy", value=train_accuracy)
        mlflow.log_metric(key="test_accuracy", value=test_accuracy)
        mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name=constants.MLFLOW_MODEL_NAME)

        # Set the model as "production"
        constants.LOGGER.info(msg=f"Send the model \"{constants.MLFLOW_MODEL_NAME}\" to production...")
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(name=constants.MLFLOW_MODEL_NAME, version=1, stage="Production")
