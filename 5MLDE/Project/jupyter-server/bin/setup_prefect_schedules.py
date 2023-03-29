# Misc.
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

# Local files
import build_model
import build_model_mlflow


def setup_schedules() -> None:
    """ Initialize prefect schedules. """
    # Define deployments
    build_model_deployment = Deployment.build_from_flow(
        name="Build local model",
        flow=build_model.main,
        version="1.0",
        tags=["training", "local"],
        schedule=(CronSchedule(cron="0 0 * * 0", timezone="Europe/Paris"))
    )

    build_model_mlflow_deployment = Deployment.build_from_flow(
        name="Build MLFlow model",
        flow=build_model_mlflow.main,
        version="1.0",
        tags=["training", "MLFlow"],
        schedule=(CronSchedule(cron="0 0 * * 0", timezone="Europe/Paris"))
    )

    # Apply them
    build_model_deployment.apply()
    build_model_mlflow_deployment.apply()


if __name__ == "__main__":
    setup_schedules()
