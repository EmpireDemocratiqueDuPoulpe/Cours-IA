#!/bin/bash

echo "Installing Python dependencies..."
python --version
python -m pip install --upgrade pip
python -m pip install --requirement /app/requirements.txt