FROM python:3.9.16-slim

# Set working directory
WORKDIR /web_service

# Create folder structure
COPY ./local_models ./local_models
COPY ./.gitignore ./.gitignore
COPY ./__init__.py ./__init__.py
COPY ./internal_config.py ./internal_config.py
COPY ./main.py ./main.py
COPY ./requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /web_service/requirements.txt

# Expose port
# > FastAPI server
EXPOSE 8001

# Run Uvicorn on container starts up
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]