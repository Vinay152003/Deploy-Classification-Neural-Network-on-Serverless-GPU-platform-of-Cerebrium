"""Utility functions for the project."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from load_mnist_local import load_mnist_from_local
from config import DATA_DIR


def load_mnist_data():
    """Load and preprocess MNIST dataset from local directory."""
    return load_mnist_from_local(DATA_DIR)


def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved to 'training_history.png'")


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_cerebrium_config(deployment_id, endpoint_url):
    """Save Cerebrium deployment configuration to file."""
    from config import CEREBRIUM_CONFIG_PATH
    
    config = {
        "deployment_id": deployment_id,
        "endpoint_url": endpoint_url
    }
    
    with open(CEREBRIUM_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Cerebrium configuration saved to {CEREBRIUM_CONFIG_PATH}")


def load_cerebrium_config():
    """Load Cerebrium deployment configuration from file."""
    from config import CEREBRIUM_CONFIG_PATH
    
    with open(CEREBRIUM_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    return config
