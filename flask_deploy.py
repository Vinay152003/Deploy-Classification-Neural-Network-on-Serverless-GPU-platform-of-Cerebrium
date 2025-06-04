from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model at startup
model_path = os.path.join('models', 'mnist_classifier.h5')
model = load_model(model_path)
print(f"Model loaded from {model_path}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    instances = np.array(data["instances"])
    
    # Ensure proper shape
    if len(instances.shape) == 3:  # Single image
        instances = instances.reshape(1, *instances.shape)
    
    # Make predictions
    predictions = model.predict(instances)
    
    # Format response
    response = {
        "predictions": predictions.tolist(),
        "predicted_classes": np.argmax(predictions, axis=1).tolist()
    }
    
    return jsonify(response)

@app.route('/', methods=['GET'])
def home():
    return "MNIST Classifier API is running. Send POST requests to /predict"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("Flask server started on port 5000")
