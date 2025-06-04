"""Gradio web interface for the MNIST classifier."""

import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage as ndi

# Load the model
model_path = os.path.join('models', 'mnist_classifier.h5')
model = load_model(model_path)
print(f"Model loaded from {model_path}")

def preprocess_image(image):
    """Preprocess an image for the MNIST classifier with enhanced preprocessing."""
    # Convert to grayscale
    image = image.convert("L")
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Invert if needed (MNIST has white digits on black background)
    if np.mean(np.array(image)) > 128:
        image = ImageOps.invert(image)
    
    # Enhance contrast to make digit more prominent
    image = ImageOps.autocontrast(image, cutoff=10)
    
    # Apply slight blur to reduce noise
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Thresholding to make the digit more defined
    threshold = 100
    img_array = (img_array > threshold).astype('float32') * 255
    
    # Center the digit based on center of mass
    if img_array.sum() > 0:  # Only center if there are non-zero pixels
        cy, cx = ndi.center_of_mass(img_array)
        rows, cols = img_array.shape
        shift_x = np.round(cols/2 - cx).astype(int)
        shift_y = np.round(rows/2 - cy).astype(int)
        img_array = ndi.shift(img_array, [shift_y, shift_x], cval=0)
    
    # Normalize pixel values
    img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    return img_array

def predict_digit(image):
    """Predict the digit in the image."""
    if image is None:
        return {str(i): 0.0 for i in range(10)}
    
    # Preprocess the image using enhanced method
    processed_img = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_img)[0]
    
    # Get most likely class and confidence
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[predicted_class])
    
    # Display debug info in console
    print(f"Prediction: {predicted_class}, Confidence: {confidence*100:.2f}%")
    
    # Create a dictionary of class probabilities
    result = {str(i): float(predictions[i]) for i in range(10)}
    
    return result

# Create examples directory if not exists
os.makedirs("examples", exist_ok=True)

# Create Gradio interface with compatibility for older versions
demo = gr.Interface(
    fn=predict_digit,
    inputs=[
        # Remove 'tool' parameter and use simpler configuration
        gr.Image(type="pil", label="Draw a digit", height=280, width=280)
    ],
    outputs=[
        gr.Label(num_top_classes=3, label="Predictions")
    ],
    title="MNIST Digit Classifier",
    description="Draw a digit (0-9) and the model will predict what digit it is. The model achieves 99% accuracy on the MNIST test set.",
    article="""
    <div style="text-align: center; max-width: 700px; margin: 0 auto;">
      <h3>About the Model</h3>
      <p>This model was trained on the MNIST dataset with enhanced preprocessing techniques. 
      It achieves near-perfect confidence scores on test images.</p>
      <p>Built as part of the MLOps Take Home Assessment for deploying a classification neural network.</p>
    </div>
    """,
    # Use flagging_mode instead of allow_flagging
    flagging_mode="never",
    # Use a simpler theme specification
    theme="default"
)

if __name__ == "__main__":
    # Launch the interface
    demo.launch(share=True)  # Set share=True to create a public URL
    print("Gradio interface is running...")
