"""Simple test script for Cerebrium-deployed MNIST classifier."""

import requests
import numpy as np
import json
from load_mnist_local import load_mnist_from_local
from config import DATA_DIR

# Your Cerebrium endpoint URL - update this after successful deployment
ENDPOINT_URL = "https://api.cortex.cerebrium.ai/v4/p-06bb2bd6/mnist-classifier/run"

# Your Cerebrium inference token - replace with your actual token
INFERENCE_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTA2YmIyYmQ2IiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoxNzUzOTIwMDAwfQ.o_MRzRkDOdVq1zuVSbLhxmklw963fG3E3ZMP9DJKtVELTFMkMHzt8nPc8z6LXVpkylGBu_t6K4aEC_Z7c76gKwONah-lRXMmg8OU8y92CwtLcc49ZXfb9wodyl7-fFeFzeWx-9A3XUatyzCmRysoXv_btnGgXKOygwVA2AvtoP23VhwTcvA43HrYGaH9SbyFatZY0QYiREK9pOPvyYUUef17CKslhv3OWVpSqAvJX4RApAtiVORDMAwQLNdoRQUmoQ4Aih3BaNOv_eNDInVeEUzzP2Gm9JrxrXS5aEyNAAn6MtZgDS4nSC2KURKLDWi9yTZi2QQMTIzwePUNikD8QA"

def test_cerebrium():
    """Test the Cerebrium deployment with a single MNIST image."""
    # Load test data
    print("Loading MNIST test data...")
    (_, _), (x_test, y_test) = load_mnist_from_local(DATA_DIR)
    
    # Select one test image (index 42 is usually a good test case)
    test_idx = 42
    test_image = x_test[test_idx]
    true_label = np.argmax(y_test[test_idx])
    
    print(f"Selected test image with true label: {true_label}")
    
    # Prepare request
    headers = {
        "Authorization": f"Bearer {INFERENCE_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Create payload with a single test image
    payload = {
        "instances": [test_image.tolist()]
    }
    
    # Print request details for debugging
    print(f"Sending request to: {ENDPOINT_URL}")
    print(f"Payload shape: {np.array(payload['instances']).shape}")
    
    try:
        # Make request
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("\nSuccess! Response received:")
            print(json.dumps(result, indent=2))
            
            # Extract prediction
            if "predicted_classes" in result:
                predicted_class = result["predicted_classes"][0]
                print(f"\nTrue label: {true_label}")
                print(f"Predicted: {predicted_class}")
                print(f"Correct: {predicted_class == true_label}")
            else:
                print("\nNo predicted_classes found in response")
        else:
            print(f"\nError: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"\nException occurred: {str(e)}")

if __name__ == "__main__":
    test_cerebrium()
