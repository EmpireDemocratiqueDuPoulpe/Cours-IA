# Local files
from config import constants
from lib.data_loading import load_data
from lib.data_preprocessing import prepare_data
from lib.modelling import train_model, get_predictions, compute_accuracy, save_pipeline


if __name__ == "__main__":
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

    # Save pipeline
    save_pipeline(
        pipeline=pipeline,
        path=(constants.LOCAL_MODELS_FOLDER / f"pipeline_v{constants.MODEL_VERSION}.joblib"),
        logger=constants.LOGGER
    )
