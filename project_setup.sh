#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Upgrade pip and install poetry
pip install --upgrade pip
pip install poetry

# Update the lock file if necessary
poetry lock

# Install dependencies and the project
poetry install

# Create and install the IPython kernel for the project
python -m ipykernel install --sys-prefix --name=mlx3129 --display-name "MLX 3.12.9"

echo "Jupyter kernel 'mlx3129' has been installed."


echo "Project setup complete!"