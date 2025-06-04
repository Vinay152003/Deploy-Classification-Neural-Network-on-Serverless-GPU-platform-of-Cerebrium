# MLOps Project: MNIST Classification with Serverless Deployment

## Project Overview
This project implements a deep learning solution for handwritten digit classification using the MNIST dataset. The model has been trained, evaluated, and deployed to allow real-time inference through multiple deployment options.

## Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Input**: 28x28 grayscale images
- **Layers**:
  - Convolutional layers with ReLU activation
  - MaxPooling layers
  - Dense layers with dropout for regularization
  - Softmax output layer for 10 classes (digits 0-9)
- **Training**: Achieved ~99% accuracy on the test set

## Preprocessing Techniques

Two preprocessing approaches were tested:

1. **Standard Preprocessing**:
   - Normalization (scaling pixel values to 0-1)

2. **Enhanced Preprocessing**:
   - Contrast enhancement
   - Gaussian blur for noise reduction
   - Thresholding
   - Centering based on center of mass

### Confidence Comparison
![Confidence Comparison](confidence_comparison.png)

The comparison shows that both preprocessing techniques achieve high confidence scores, with the enhanced method providing slight improvements in edge cases.

## Deployment Methods

### 1. Flask Web API
A Python Flask application that serves predictions through a REST API endpoint. This approach is lightweight and suitable for local or small-scale deployments.

### 2. Gradio Interactive UI
A user-friendly web interface that allows drawing digits and receiving predictions in real-time. This is ideal for demonstrations and user testing.

### 3. Alternative Deployment Options (Considered)
- **Cerebrium Serverless GPU**: Cloud-based serverless deployment for GPU acceleration
- **Docker Containerization**: Packaging the model with its dependencies for consistent deployment
- **TensorFlow Serving**: Production-grade serving system designed for TensorFlow models

## Model Performance
- **Accuracy**: 99.04% on the test set
- **Confidence**: Near 100% confidence on correctly classified examples
- **Inference Speed**: < 100ms per prediction on CPU

## Conclusion

The MNIST classifier demonstrates excellent accuracy and confidence in its predictions. The enhanced preprocessing technique provides a small but valuable improvement in confidence scores for certain input variations.

The multiple deployment options showcase flexibility in serving machine learning models, from simple local APIs to interactive web interfaces, aligning with the MLOps requirement of supporting different operational scenarios.
