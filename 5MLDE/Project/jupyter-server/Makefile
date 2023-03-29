.DEFAULT_GOAL = help

.PHONY: init_model_mlflow
init_model_mlflow:
	@echo "Installing system dependencies..."
	@apt-get install curl -y
	@echo "Installing Python dependencies..."
	@python -m pip install -r requirements.txt
	@echo "Building local model..."
	@python ./bin/build_model.py
	@echo "Building model and pushing to Mlflow..."
	@python ./bin/build_model_mlflow.py
	@echo "We've done our job. It works (normally)."

.PHONY: help
help:
	@echo
	@echo "Welcome to the file that deploy our awesome model to MLFlow!"
	@echo
	@echo "Available commands:"
	@echo
	@echo "make init_model_mlflow"
	@echo "	- Prepare the model and push it as production to MLFlow."
	@echo
	@echo "make help"
	@echo "	- Show this help message."
	@echo