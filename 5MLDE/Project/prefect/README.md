# Prefect
Prefect is an orchestrator. It can transform any Python function into a unit of work that can be observed from a web interface.

## Installation - last step
_Unless otherwise specified, the current working directory should be the project root for all commands to be run on the host PC._

1. From the Jupyter container, run the following commands:
    1. Set the API URL: `prefect config set PREFECT_API_URL=http://localhost:4200/api`
    2. Start the prefect server: `prefect server start --host 0.0.0.0`
2. Open the following URI in your favorite browser: `http://localhost:4200`
3. Open another terminal in the Jupyter container and run the following commands:
   1. Move to the project folder: `cd /app`
   2. Test the Prefect flow: `python ./bin/build_model.py`
   3. Schedule automatic deployment: `python ./bin/setup_prefect_schedules.py`

You should now see the flows running in the Prefect interface.

## Navigation
If this installation works, you have successfully set up the whole infrastructure.