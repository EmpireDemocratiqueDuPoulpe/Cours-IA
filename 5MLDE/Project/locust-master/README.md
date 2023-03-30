# Locust - Master node
Locust is a scalable load testing framework written in Python. It will be very useful for testing our API and system performance.

## Installation - step five
_Unless otherwise specified, the current working directory should be the project root for all commands to be run on the host PC._

1. From your host computer, run the following commands:
    1. Build the Docker image: `docker build --tag 5mlde_locust ./locust-master`
    2. Run the container: `docker run --interactive --tty --user root --publish 8089:8089 --network 5mlde-network --name locust-master --detach 5mlde_locust`
2. Open the following URI in your favorite browser: `http://localhost:8089`

You can now enter the number of users you want and the delay between each spawn. Set the host URI to `http://prediction-server:8001` and start the test.

## Navigation
If the following installation works, you can proceed to the last step: [starting Prefect](https://github.com/EmpireDemocratiqueDuPoulpe/Cours-IA/tree/main/5MLDE/Project/prefect).