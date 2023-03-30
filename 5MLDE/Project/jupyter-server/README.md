# Jupyter server
The jupyter server is the core of the environment. This allows us to develop our model and then send it to MLFlow and other services.

## Installation - step one
__Unless otherwise specified, the current working directory should be the project root for all commands to be run on the host PC.__

1. Transfert the [Android Malware dataset](https://www.kaggle.com/datasets/subhajournal/android-malware-detection) to the [./data](https://github.com/EmpireDemocratiqueDuPoulpe/Cours-IA/tree/main/5MLDE/Project/jupyter-server/data) folder. It was not included because the file is too large.
2. From your host computer, run the following commands:
   1. Build the Docker image: `docker build --tag 5mlde_jupyter ./jupyter-server`
   2. Run the container: `docker run --interactive --tty --user root --publish 10000:8888 --publish 8000:8000 --publish 4200:4200 --volume ${PWD}/mlflow-server/local:/mlflow --env JUPYTER_ENABLE_LAB=yes --env MLFLOW_TRACKING_URI=http://mlflow:5000 --network 5mlde-network --name jupyter --detach 5mlde_jupyter`
3. Open the following URI in your favorite browser: `http://localhost:10000`

You can now explore the projet structure and read the notebooks about the data exploration and validation in [./notebooks](https://github.com/EmpireDemocratiqueDuPoulpe/Cours-IA/tree/main/5MLDE/Project/jupyter-server/notebooks).

Other services might ask you to run a command in this Docker container. To do this, open a terminal from the Jupyter launcher.

## Navigation
If the following installation works, you can proceed to the next step: [installing MLFlow](https://github.com/EmpireDemocratiqueDuPoulpe/Cours-IA/tree/main/5MLDE/Project/mlflow-server).