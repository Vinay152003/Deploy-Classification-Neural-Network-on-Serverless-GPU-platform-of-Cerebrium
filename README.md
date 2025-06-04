# MNIST Classification - MLOps Take Home Assessment

This project demonstrates deploying a classification neural network on various platforms, including local deployment and serverless options.

## Project Structure

- **config.py**: Configuration parameters
- **load_mnist_local.py**: Code to load the MNIST dataset from local directory
- **train_model.py**: Script to train the classification model
- **utils.py**: Utility functions
- **test_model_confidence.py**: Script to test model confidence with different preprocessing methods
- **flask_deploy.py**: Deploy the model as a Flask API
- **gradio_interface.py**: Interactive web interface using Gradio
- **deployment_options.md**: Documentation of various deployment methods

## Setup Instructions

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model (if not already trained):
   ```
   python train_model.py
   ```

3. Test model confidence:
   ```
   python test_model_confidence.py
   ```

4. Deploy with Flask:
   ```
   python flask_deploy.py
   ```

5. Run the Gradio interface:
   ```
   python gradio_interface.py
   ```

## Model Performance

The MNIST classifier achieves outstanding performance:
- **Accuracy**: 99.04% on the test set
- **Confidence**: Near 100% on most test examples
- **Enhanced Preprocessing**: Improves confidence on edge cases

![Confidence Comparison](confidence_comparison.png)

### Gradio Interface Results

When testing with real user-uploaded images in the Gradio interface:
- Successfully classified a digit "8" with 46% confidence
- Secondary predictions included "3" (25%) and "5" (13%)
- The model correctly identified the digit despite variations in handwriting style

This real-world testing demonstrates the model's ability to generalize beyond the MNIST dataset, though with lower confidence compared to the test set. This is expected when processing user-drawn or uploaded digits that differ from the training data distribution.

## Deployment Results

Multiple deployment options have been implemented:
- **Local Flask API**: Accessible at http://127.0.0.1:5000/predict
- **Interactive Gradio UI**: Provides a user-friendly drawing interface
- **Deployment documentation**: Additional options in deployment_options.md

## Project Report

See [project_report.md](project_report.md) for a detailed analysis of the model's performance and deployment strategies.

## Future Improvements

- Implement additional data augmentation for improved robustness
- Explore model optimization techniques like quantization
- Implement a CI/CD pipeline for automated deployment
