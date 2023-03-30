# MLFlow server
MLFlow is an open source platform used to manage the Machine Learning lifecycle. It permits reproducibility by recording experiments, configuration, parameters used in a model training.

## Installation - step three
_Unless otherwise specified, the current working directory should be the project root for all commands to be run on the host PC._

1. From your host computer, run the following commands:
   1. Build the Docker image: `docker build --tag 5mlde_mlflow ./mlflow-server`
   2. Run the container: `docker run --interactive --tty --user root --publish 5000:5000 --volume ${PWD}/mlflow-server/local:/mlflow --network 5mlde-network --name mlflow --detach 5mlde_mlflow`
2. Open the following URI in your favorite browser: `http://localhost:5000`
3. From the Jupyter container, run the following commands:
   1. Move to the project folder: `cd /app`
   2. Initialize the Machine Learning model: `make init_model_mlflow`

You can reload the MLFlow webpage and the fitted model should appear in the list on the left.

## Navigation
If this installation works, you can proceed to the next step: [installing the prediction server](https://github.com/EmpireDemocratiqueDuPoulpe/Cours-IA/tree/main/5MLDE/Project/prediction-server).