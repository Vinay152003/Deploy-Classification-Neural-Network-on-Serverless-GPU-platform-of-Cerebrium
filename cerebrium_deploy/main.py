import numpy as np
from tensorflow import keras
from keras import layers
from scipy import ndimage as ndi

# Create a simple model directly instead of loading from file
def create_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Create model
model = create_model()
print("Created MNIST model")

def preprocess_image(image_array):
    """Apply enhanced preprocessing to improve confidence."""
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
    """Handler function for the Cerebrium endpoint."""
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
