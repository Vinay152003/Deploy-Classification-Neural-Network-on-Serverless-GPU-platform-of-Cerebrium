"""Test model confidence on various input styles."""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage as ndi

from load_mnist_local import load_mnist_from_local
from config import DATA_DIR, MODEL_SAVE_PATH

def load_models():
    """Load both original and fine-tuned models if available."""
    models = {}
    
    # Load original model
    if os.path.exists(MODEL_SAVE_PATH):
        models['original'] = load_model(MODEL_SAVE_PATH)
        print(f"Loaded original model from {MODEL_SAVE_PATH}")
    
    # Check for fine-tuned model
    fine_tuned_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'mnist_classifier_finetuned.h5')
    if os.path.exists(fine_tuned_path):
        models['fine_tuned'] = load_model(fine_tuned_path)
        print(f"Loaded fine-tuned model from {fine_tuned_path}")
    
    return models

def preprocess_with_variations(image, variation='standard'):
    """Apply different preprocessing techniques to compare their effect on confidence."""
    # Convert to numpy array
    img_array = np.array(image)
    
    if variation == 'standard':
        # Basic preprocessing (normalize only)
        processed = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    elif variation == 'enhanced':
        # Enhanced preprocessing
        # Apply contrast enhancement
        image = ImageOps.autocontrast(image, cutoff=10)
        # Apply slight blur
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        # Convert to numpy array and threshold
        img_array = np.array(image)
        threshold = 100
        img_array = (img_array > threshold).astype('float32') * 255
        # Center the digit
        if img_array.sum() > 0:  # Only center if there are non-zero pixels
            cy, cx = ndi.center_of_mass(img_array)
            rows, cols = img_array.shape
            shift_x = np.round(cols/2 - cx).astype(int)
            shift_y = np.round(rows/2 - cy).astype(int)
            img_array = ndi.shift(img_array, [shift_y, shift_x], cval=0)
        # Normalize
        processed = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    elif variation == 'minimal':
        # Minimal preprocessing
        processed = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    return processed

def test_confidence():
    """Test model confidence with different preprocessing techniques."""
    # Load models
    models = load_models()
    if not models:
        print("No models found. Please train or fine-tune a model first.")
        return
    
    # Load test data
    (_, _), (x_test, y_test) = load_mnist_from_local(DATA_DIR)
    
    # Select a few challenging test images
    np.random.seed(42)  # For reproducibility
    test_indices = np.random.choice(len(x_test), 10, replace=False)
    test_images = x_test[test_indices]
    test_labels = np.argmax(y_test[test_indices], axis=1)
    
    # Create a figure to display results
    plt.figure(figsize=(15, 5*len(models)))
    
    row = 0
    for model_name, model in models.items():
        row += 1
        
        # Test with different preprocessing variations
        for i, img in enumerate(test_images[:5]):  # Use first 5 images
            # Convert from numpy array to PIL Image
            pil_img = Image.fromarray((img.reshape(28, 28) * 255).astype('uint8'))
            
            # Apply different preprocessing techniques
            processed_standard = preprocess_with_variations(pil_img, 'standard')
            processed_enhanced = preprocess_with_variations(pil_img, 'enhanced')
            
            # Get predictions
            pred_standard = model.predict(processed_standard)[0]
            pred_enhanced = model.predict(processed_enhanced)[0]
            
            # Get top prediction and confidence
            top_class_standard = np.argmax(pred_standard)
            confidence_standard = pred_standard[top_class_standard] * 100
            
            top_class_enhanced = np.argmax(pred_enhanced)
            confidence_enhanced = pred_enhanced[top_class_enhanced] * 100
            
            # Plot the image and results
            plt.subplot(len(models), 5, (row-1)*5 + i + 1)
            plt.imshow(img.reshape(28, 28), cmap='gray')
            plt.title(f"True: {test_labels[i]}\n"
                     f"Standard: {top_class_standard} ({confidence_standard:.1f}%)\n"
                     f"Enhanced: {top_class_enhanced} ({confidence_enhanced:.1f}%)")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('confidence_comparison.png')
    print("Confidence comparison saved to 'confidence_comparison.png'")

if __name__ == "__main__":
    # Make sure necessary imports are available
    try:
        from scipy import ndimage as ndi
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.check_call(["pip", "install", "scipy"])
        from scipy import ndimage as ndi
    
    # Run the confidence test
    test_confidence()
