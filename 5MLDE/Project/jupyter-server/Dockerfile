FROM jupyter/datascience-notebook

# Set working directory
WORKDIR /app

# Create folder structure
COPY ./bin ./bin
COPY ./config ./config
COPY ./data ./data
COPY ./helpers ./helpers
COPY ./lib ./lib
COPY ./models ./models
COPY ./notebooks ./notebooks
COPY ./temp ./temp
COPY ./.gitignore ./.gitignore
COPY ./Makefile ./Makefile
COPY ./requirements.txt ./requirements.txt

# Install system dependencies
USER root
RUN apt update
RUN apt install build-essential -y

# Install Python dependencies
RUN bash ./bin/install_requirements.sh
ENV PYTHONPATH "$PYTHONPATH:/app/lib"

# Expose port
# > Jupyter Lab
EXPOSE 8888
# > Deployed model server (internal. Not to confuse with the 8001 port of the standalone server)
EXPOSE 8000
# > Prefect
EXPOSE 4200

# Run Jupyter Lab on container starts up
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token='5MLDE'"]