"""Client to test the locally deployed Flask API for MNIST classification."""

import requests
import numpy as np
import matplotlib.pyplot as plt
from load_mnist_local import load_mnist_from_local
from config import DATA_DIR

def test_flask_api(url="http://127.0.0.1:5000/predict"):
    """Test the Flask API with sample MNIST images."""
    print(f"Testing MNIST classifier at: {url}")
    
    # Load test data
    print("Loading MNIST test data...")
    (_, _), (x_test, y_test) = load_mnist_from_local(DATA_DIR)
    
    # Select a few test images
    num_samples = 5
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    test_images = x_test[indices]
    test_labels = np.argmax(y_test[indices], axis=1)  # Convert from one-hot to digit
    
    # Prepare the request payload
    payload = {
        "instances": test_images.tolist()
    }
    
    # Set up headers
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Sending request with {num_samples} test images...")
    
    # Make the request
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            predicted_classes = result.get("predicted_classes", [])
            
            print("Predictions received successfully!")
            
            # Display results
            plt.figure(figsize=(15, 3))
            for i in range(num_samples):
                plt.subplot(1, num_samples, i+1)
                plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
                plt.title(f"True: {test_labels[i]}\nPred: {predicted_classes[i]}")
                plt.axis('off')
            
            plt.savefig('flask_inference_results.png')
            print(f"Results saved to 'flask_inference_results.png'")
            
            # Calculate accuracy
            correct = sum(1 for i in range(num_samples) if predicted_classes[i] == test_labels[i])
            accuracy = correct / num_samples
            print(f"Accuracy on sample: {accuracy:.2f} ({correct}/{num_samples})")
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error connecting to Flask API: {str(e)}")
        print("\nMake sure the Flask server is running at http://127.0.0.1:5000")

if __name__ == "__main__":
    test_flask_api()
