FROM jupyter/datascience-notebook

# Set working directory
WORKDIR /app

# Create folder structure
COPY ./bin ./bin
COPY ./config ./config
COPY ./data ./data
COPY ./helpers ./helpers
COPY ./lib ./lib
COPY ./locust ./locust
COPY ./models ./models
COPY ./notebooks ./notebooks
COPY ./temp ./temp
COPY ./web_service ./web_service
COPY ./.gitignore ./.gitignore
COPY ./requirements.txt ./requirements.txt

# Install Python dependencies
RUN bash ./bin/install_requirements.sh
ENV PYTHONPATH "$PYTHONPATH:/app/lib"

# Expose port
# > Jupyter Lab
EXPOSE 8888
# > Deployed model server
EXPOSE 8000
# > Prefect
EXPOSE 4200

# Run Jupyter Lab on container starts up
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token='5MLRE'"]