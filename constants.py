"""All project global constants."""

import os

PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
TENSORBOARD_LOGS_PATH = os.path.join(PROJECT_ROOT_PATH, "tensorboardruns")
MLFLOW_BACKEND_STORE_PATH = os.path.join(PROJECT_ROOT_PATH, "mlruns")
