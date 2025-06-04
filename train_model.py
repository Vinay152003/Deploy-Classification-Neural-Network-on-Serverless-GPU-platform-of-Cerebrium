"""Train a classification neural network for MNIST dataset."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam

from utils import load_mnist_data, plot_training_history, ensure_dir
from config import (
    MODEL_NAME, 
    INPUT_SHAPE, 
    NUM_CLASSES, 
    BATCH_SIZE, 
    EPOCHS, 
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_SAVE_PATH
)


def build_model():
    """Build a convolutional neural network for image classification."""
    model = Sequential([
        # Convolutional layer
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional layer
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_and_save_model():
    """Train the model and save it."""
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Build the model
    model = build_model()
    
    # Print model summary
    model.summary()
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_split=0.1
    )
    
    # Evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    ensure_dir(MODEL_DIR)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    train_and_save_model()
