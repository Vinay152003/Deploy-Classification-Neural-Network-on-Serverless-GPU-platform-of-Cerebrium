# MNIST Classification Neural Network Deployment

This project demonstrates the deployment of a classification neural network for handwritten digit recognition on various platforms, including Cerebrium's serverless GPU platform.

## Project Overview

- Trained a CNN model on the MNIST dataset achieving 99.04% accuracy
- Implemented preprocessing techniques that improve confidence on real-world inputs
- Deployed the model via multiple methods:
  - Flask REST API for local deployment
  - Gradio web interface for interactive testing
  - Cerebrium serverless platform for cloud deployment

## Repository Structure

## Files Included in This Repository

Based on the project requirements, the following files are included in this submission:

### Core Python Files
- `config.py` - Configuration parameters
- `load_mnist_local.py` - Loads MNIST dataset from local directory
- `train_model.py` - Trains the classification neural network
- `utils.py` - Utility functions
- `requirements.txt` - Dependencies list

### Testing and Evaluation
- `test_model_confidence.py` - Tests model confidence with different preprocessing methods
- `test_flask_client.py` - Client script to test Flask API
- `test_inference.py` - Script to test model inference
- `confidence_comparison.png` - Visual comparison of preprocessing techniques

### Deployment Files
- Flask Deployment:
  - `flask_deploy.py` - Local Flask API deployment

- Gradio Interface:
  - `gradio_interface.py` - Interactive web interface

- Dashboard Cerebrium.ai Screenshot
![image](https://github.com/user-attachments/assets/9b53fee7-6c43-4dea-8022-867bf0a74dff)

- Cerebrium Deployment:
  - `cerebrium_deploy/main.py` - Cerebrium handler function
  - `cerebrium_deploy/cerebrium.toml` - Cerebrium configuration
  - `deploy_cerebrium.py` - Original deployment script
  - `setup_cerebrium_deployment.py` - Helper script for setup
  - `simple_cerebrium_test.py` - Simplified test script

### Model Files
- `models/mnist_classifier.h5` - The trained model file

### Other Essential Files
- `.gitignore` - Specifies files to ignore in Git
- `training_history.png` - Training history visualization

Note: The `.env` file and `__pycache__` directories are not included as they contain sensitive information or temporary files.

## Setup and Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the desired deployment method:
   - Flask API: `python flask_deploy.py`
   - Gradio interface: `python gradio_interface.py`

## Model Performance

The MNIST classifier achieves:
- **Accuracy**: 99.04% on the test set
- **Confidence**: Near 100% on most test examples
- **Enhanced Preprocessing**: Improves confidence on edge cases

![Confidence Comparison](![confidence_comparison](https://github.com/user-attachments/assets/836b6b4c-fd4e-4317-892c-015a6a2fdfef)
)

## Deployment Results

### Gradio Interface
The Gradio interface provides an interactive way to test the model with hand-drawn or uploaded digits.
![image](https://github.com/user-attachments/assets/35bca163-c704-468c-a12d-346a414af877)


### Flask API
The Flask API provides a REST endpoint for making predictions, accessible at http://127.0.0.1:5000/predict

### Cerebrium Deployment
The model was deployed to Cerebrium's serverless platform, allowing for scalable, cloud-based inference.

## Challenges and Solutions

- **Version Compatibility**: Resolved TensorFlow version incompatibilities between local environment and Cerebrium by defining the model architecture directly in the deployment code
- **Preprocessing Optimization**: Improved prediction confidence by implementing enhanced preprocessing techniques, including thresholding and centering

## Future Improvements

- Implement additional data augmentation for improved robustness
- Explore model optimization techniques like quantization
- Implement CI/CD pipeline for automated deployment

## Author

Vinay Hipparge

## Acknowledgments

- MTailor for providing this assessment challenge
- Cerebrium for their serverless ML platform
