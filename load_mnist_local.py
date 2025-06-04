"""Load MNIST dataset from local directory in IDX format."""

import os
import numpy as np
from tensorflow.keras.utils import to_categorical

def read_idx_file(filename):
    """
    Read IDX file format according to the MNIST specification.
    
    Format specification: http://yann.lecun.com/exdb/mnist/
    """
    with open(filename, 'rb') as f:
        # First 4 bytes contain the magic number
        magic = int.from_bytes(f.read(4), byteorder='big')
        
        # The 4th byte (least significant) contains the number of dimensions
        ndim = magic & 0xFF
        
        # Read the dimensions (4 bytes per dimension)
        dims = []
        for _ in range(ndim):
            dims.append(int.from_bytes(f.read(4), byteorder='big'))
        
        # Read the data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Reshape the data according to the dimensions
        data = data.reshape(dims)
        
        return data

def load_mnist_from_local(data_dir):
    """
    Load MNIST dataset from local directory in IDX format.
    
    Args:
        data_dir: Path to directory containing MNIST IDX files
    
    Returns:
        (x_train, y_train), (x_test, y_test): Tuple of training and test data
    """
    print(f"Loading MNIST data from: {data_dir}")
    
    # Define file paths
    train_images_file = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_labels_file = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_images_file = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_file = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
    
    # Check if files exist
    for file_path in [train_images_file, train_labels_file, test_images_file, test_labels_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
    
    # Load data from IDX files
    print("Loading training images...")
    x_train = read_idx_file(train_images_file)
    print("Loading training labels...")
    y_train = read_idx_file(train_labels_file)
    print("Loading test images...")
    x_test = read_idx_file(test_images_file)
    print("Loading test labels...")
    y_test = read_idx_file(test_labels_file)
    
    # Reshape and normalize images
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"Loaded {x_train.shape[0]} training samples and {x_test.shape[0]} test samples")
    
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    # Test loading data
    data_path = r"C:\Users\vrhso\Downloads\archive (1)"
    (x_train, y_train), (x_test, y_test) = load_mnist_from_local(data_path)
    
    # Print dataset information
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Display sample distribution
    digit_counts = [sum(np.argmax(y_train, axis=1) == i) for i in range(10)]
    print("\nTraining dataset digit distribution:")
    for digit, count in enumerate(digit_counts):
        print(f"Digit {digit}: {count} samples")
