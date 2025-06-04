"""Test the deployed model on Cerebrium."""

import os
import json
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from load_mnist_local import load_mnist_from_local
from utils import load_cerebrium_config
from config import DATA_DIR


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Check if Cerebrium inference token is set
    inference_token = os.environ.get("CEREBRIUM_INFERENCE_TOKEN")
    if not inference_token:
        raise ValueError("CEREBRIUM_INFERENCE_TOKEN environment variable not set")
    return inference_token


def update_endpoint_url():
    """Prompt user to provide the correct endpoint URL."""
    print("The current endpoint URL seems to be incorrect.")
    print("Please provide the correct endpoint URL from the Cerebrium dashboard:")
    new_url = input("Enter the correct endpoint URL: ")
    
    if new_url:
        # Load and update the config
        try:
            with open("cerebrium_config.json", "r") as f:
                config = json.load(f)
            
            config["endpoint_url"] = new_url
            
            with open("cerebrium_config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            print(f"Updated endpoint URL to: {new_url}")
            return new_url
        except Exception as e:
            print(f"Failed to update config: {str(e)}")
            return None
    return None


def test_cerebrium_deployment():
    """Test the Cerebrium deployment with sample data."""
    # Load environment variables
    inference_token = load_environment()
    
    # Load MNIST test data
    (_, _), (x_test, y_test) = load_mnist_from_local(DATA_DIR)
    
    # Load Cerebrium configuration
    config = load_cerebrium_config()
    endpoint_url = config["endpoint_url"]
    
    print(f"Testing model at endpoint: {endpoint_url}")
    print("Checking if endpoint is reachable...")
    
    # Test endpoint availability
    try:
        response = requests.head(endpoint_url, timeout=5)
        print(f"Endpoint status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Endpoint is not reachable: {str(e)}")
        new_url = update_endpoint_url()
        if new_url:
            endpoint_url = new_url
        else:
            print("\nPossible solutions:")
            print("1. Make sure your model is correctly deployed on Cerebrium")
            print("2. Check if you're using the correct endpoint URL")
            print("3. Try the following alternative endpoint formats:")
            print(f"   - https://api.cerebrium.ai/v2/p-06bb2bd6/{CEREBRIUM_MODEL_NAME}/predict")
            print(f"   - https://api.cerebrium.ai/v1/p-06bb2bd6/{CEREBRIUM_MODEL_NAME}/predict")
            print("4. Check Cerebrium documentation for the correct endpoint format")
            return
    
    # Select a few test images
    num_samples = 5
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    test_images = x_test[indices]
    test_labels = np.argmax(y_test[indices], axis=1)  # Convert from one-hot to digit
    
    # Create the request payload
    payload = {
        "instances": test_images.tolist()
    }
    
    # Alternative payload formats to try if the first one fails
    alternative_payloads = [
        {"inputs": {"instances": test_images.tolist()}},
        {"data": test_images.tolist()},
        {"images": test_images.tolist()}
    ]
    
    # Set up headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {inference_token}"
    }
    
    print(f"Sending request with {num_samples} test images...")
    
    # Make the request
    start_time = time.time()
    response = None
    error = None
    
    # Try the main payload format
    try:
        response = requests.post(endpoint_url, json=payload, headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        error = str(e)
        response = None
    
    # If the main request failed, try alternative payload formats
    if response is None or response.status_code != 200:
        print("Main request failed. Trying alternative payload formats...")
        for i, alt_payload in enumerate(alternative_payloads):
            try:
                print(f"Trying alternative payload format {i+1}...")
                response = requests.post(endpoint_url, json=alt_payload, headers=headers, timeout=10)
                if response.status_code == 200:
                    print(f"Alternative payload format {i+1} succeeded!")
                    payload = alt_payload  # Remember the successful format
                    break
            except requests.exceptions.RequestException as e:
                error = str(e)
                continue
    
    end_time = time.time()
    latency = end_time - start_time
    
    # Process the response
    if response and response.status_code == 200:
        result = response.json()
        print(f"Response received in {latency:.4f} seconds")
        print(f"Response content: {result}")
        
        # Try to extract predicted classes based on different response formats
        predicted_classes = None
        if "predicted_classes" in result:
            predicted_classes = result["predicted_classes"]
        elif "predictions" in result:
            # Try to convert raw predictions to classes
            predictions = np.array(result["predictions"])
            predicted_classes = np.argmax(predictions, axis=1).tolist()
        elif isinstance(result, list) and len(result) == num_samples:
            # The response might be a list of predictions
            predicted_classes = [np.argmax(pred) if isinstance(pred, list) else pred for pred in result]
        
        if predicted_classes:
            # Display results
            plt.figure(figsize=(15, 3))
            for i in range(num_samples):
                plt.subplot(1, num_samples, i+1)
                plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
                plt.title(f"True: {test_labels[i]}\nPred: {predicted_classes[i]}")
                plt.axis('off')
            
            plt.savefig('inference_results.png')
            print(f"Results saved to 'inference_results.png'")
            
            # Calculate accuracy
            correct = sum(1 for i in range(num_samples) if predicted_classes[i] == test_labels[i])
            accuracy = correct / num_samples
            print(f"Accuracy on sample: {accuracy:.2f} ({correct}/{num_samples})")
        else:
            print("Could not extract predictions from response. Response format:")
            print(json.dumps(result, indent=2))
    else:
        error_msg = response.text if response else error
        status_code = response.status_code if response else "No response"
        print(f"Error: {status_code}")
        print(f"Error details: {error_msg}")
        
        print("\nTroubleshooting steps:")
        print("1. Verify your model has been deployed on Cerebrium")
        print("2. Check that the endpoint URL is correct")
        print("3. Ensure your inference token is valid")
        print("4. Try a different payload format")
        print("5. Check Cerebrium documentation for the correct request format")


if __name__ == "__main__":
    test_cerebrium_deployment()
