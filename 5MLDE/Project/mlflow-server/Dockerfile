FROM ghcr.io/mlflow/mlflow:v2.2.2

# Set working directory
WORKDIR /mlflow

# Set environment variables
ENV BACKEND_URI sqlite:////mlflow/mlflow.db
ENV ARTIFACT_ROOT /mlflow/artifacts
ENV ARTIFACT_DESTINATION /mlflow/artifacts

# Expose port
# > MLFlow server
EXPOSE 5000

# Run MLFlow on container starts up
CMD mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --artifacts-destination ${ARTIFACT_DESTINATION} --serve-artifacts --host 0.0.0.0 --port 5000