# Prediction server
Now that the model is built and ready, we will configure an API to serve it on the local network.

## Installation - step four
_Unless otherwise specified, the current working directory should be the project root for all commands to be run on the host PC._

1. From your host computer, run the following commands:
    1. Build the Docker image: `docker build --tag 5mlde_prediction-server ./prediction-server`
    2. Run the container: `docker run --interactive --tty --user root --publish 8001:8001 --volume ${PWD}/mlflow-server/local:/mlflow --network 5mlde-network --name prediction-server --detach 5mlde_prediction-server`
2. Open the following URI in your favorite browser: `http://localhost:8001`
3. To test it further, run the following commands from the Jupyter container:
   1. Move to the project folder: `cd /app`
   2. Send a test request to the API: `python ./bin/post_payload.py`

The API is has three pages:
- `http://localhost:8001` - The "home" page.
- `http://localhost:8001/predict` - The prediction endpoint.
- `http://localhost:8001/docs` - A more intuitive way to test the endpoints.

## Navigation
If this installation works, you can proceed to the next step: [installing Locust](https://github.com/EmpireDemocratiqueDuPoulpe/Cours-IA/tree/main/5MLDE/Project/locust-master).