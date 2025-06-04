"""Configuration parameters for the project."""

import os

# Local data parameters
DATA_DIR = r"C:\Users\vrhso\Downloads\archive (1)"

# Model parameters
MODEL_NAME = "mnist_classifier"
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 5  # Reduced for faster training
LEARNING_RATE = 0.001

# Cerebrium deployment parameters
CEREBRIUM_MODEL_NAME = "mnist-classifier"
CEREBRIUM_MODEL_VERSION = "v1"
HARDWARE_TYPE = "cpu.small"  # or "gpu.small" for GPU acceleration
TIMEOUT = 30  # seconds

# Paths
MODEL_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "mnist_classifier.h5")
CEREBRIUM_CONFIG_PATH = "cerebrium_config.json"
