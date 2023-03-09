from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

import orchestration_03_machine_learning_workflow as flows


#### Schedule setup ################################################################
def setup_schedules() -> None:
    # Define deployments
    complete_ml_deployment = Deployment.build_from_flow(
        name="run_training",
        flow=flows.complete_ml,
        version="1.0",
        tags=["training"],
        schedule=(CronSchedule(cron="0 0 * * 0", timezone="Europe/Paris"))
    )
    
    batch_inference_deployment = Deployment.build_from_flow(
        name="run_predictions",
        flow=flows.batch_inference,
        version="1.0",
        tags=["inference"],
        schedule=(CronSchedule(cron="0 * * * *", timezone="Europe/Paris"))
    )
    
    # Apply them
    complete_ml_deployment.apply()
    batch_inference_deployment.apply()


#### Main ##########################################################################
if __name__ == "__main__":
    setup_schedules()
