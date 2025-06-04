"""
Helper script to set up Cerebrium deployment files.
This script creates the necessary directory structure and files for deploying
the MNIST classifier to Cerebrium.
"""

import os
import shutil
import sys

def create_cerebrium_deployment():
    """Create the Cerebrium deployment directory and files."""
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models")
    model_path = os.path.join(model_dir, "mnist_classifier.h5")
    cerebrium_dir = os.path.join(current_dir, "cerebrium_deploy")
    
    # Create directory
    print(f"Creating directory: {cerebrium_dir}")
    os.makedirs(cerebrium_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running: python train_model.py")
        return False
    
    # Copy model file
    print(f"Copying model from {model_path} to {cerebrium_dir}")
    shutil.copy(model_path, os.path.join(cerebrium_dir, "mnist_classifier.h5"))
    
    # Create main.py
    main_py_path = os.path.join(cerebrium_dir, "main.py")
    print(f"Creating {main_py_path}")
    with open(main_py_path, "w") as f:
        f.write("""import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy import ndimage as ndi

# Load model at startup
model = load_model('mnist_classifier.h5')
print("MNIST model loaded successfully")

def preprocess_image(image_array):
    \"\"\"Apply enhanced preprocessing to improve confidence.\"\"\"
    # Convert to numpy array
    img_array = np.array(image_array)
    
    # Apply thresholding to make digit more defined
    threshold = 100
    img_array = (img_array > threshold).astype('float32') * 255
    
    # Center the digit based on center of mass
    if img_array.sum() > 0:
        cy, cx = ndi.center_of_mass(img_array)
        rows, cols = img_array.shape
        shift_x = np.round(cols/2 - cx).astype(int)
        shift_y = np.round(rows/2 - cy).astype(int)
        img_array = ndi.shift(img_array, [shift_y, shift_x], cval=0)
    
    # Normalize pixel values
    img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    return img_array

def run(instances):
    \"\"\"Handler function for the Cerebrium endpoint.\"\"\"
    try:
        # Process inputs (expecting normalized images)
        instances_array = np.array(instances)
        
        # Ensure proper shape
        if len(instances_array.shape) == 3:  # Single image
            instances_array = instances_array.reshape(1, *instances_array.shape)
        
        # Make predictions
        predictions = model.predict(instances_array)
        
        # Format response
        response = {
            "predictions": predictions.tolist(),
            "predicted_classes": np.argmax(predictions, axis=1).tolist()
        }
        
        return response
        
    except Exception as e:
        return {"error": str(e)}
""")
    
    # Create cerebrium.toml
    toml_path = os.path.join(cerebrium_dir, "cerebrium.toml")
    print(f"Creating {toml_path}")
    with open(toml_path, "w") as f:
        f.write("""[cerebrium.deployment]
name = "mnist-classifier"
python_version = "3.9"
include = ["./*", "main.py", "cerebrium.toml", "mnist_classifier.h5"]
exclude = [".*", "__pycache__/", "venv/"]

[cerebrium.hardware]
cpu = 2
memory = 4.0
compute = "CPU"  # Use "GPU" if GPU acceleration is needed

[cerebrium.scaling]
min_replicas = 0
max_replicas = 2
cooldown = 10

[cerebrium.dependencies.pip]
tensorflow = "2.13.0"
numpy = "1.24.3"
scipy = "1.10.1"
pillow = "9.5.0"
""")
    
    print("\nCerebrium deployment files created successfully!")
    print(f"Files created in: {cerebrium_dir}")
    print("\nNext steps:")
    print("1. Install Cerebrium CLI: pip install cerebrium")
    print("2. Navigate to the deployment directory: cd", cerebrium_dir)
    print("3. Log in to Cerebrium: cerebrium login")
    print("4. Deploy your model: cerebrium deploy")
    
    return True

if __name__ == "__main__":
    create_cerebrium_deployment()
