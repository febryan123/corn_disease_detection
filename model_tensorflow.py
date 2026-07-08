import os
import numpy as np
from PIL import Image
import io

# Optional: Disable TF warnings if needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
except ImportError:
    tf = None

# Define the classes based on alphabetical folder order used by Keras flow_from_directory
CLASSES = [
    "Bercak_Daun",
    "Hawar_Daun",
    "Karat_Daun",
    "Sehat"
]

def load_model(weights_path="models/model_final.h5"):
    """
    Load TensorFlow/Keras model.
    """
    if tf is None:
        print("TensorFlow is not installed. Please install it using: pip install tensorflow")
        return None
        
    if os.path.exists(weights_path):
        model = tf.keras.models.load_model(weights_path)
        print(f"Loaded model from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found. Please place your trained model here.")
        model = None
        
    return model

def predict_image(image_bytes, model):
    """
    Process an image and return the predicted class and confidence.
    """
    if model is None:
         return {
            "success": False,
            "error": "Model not loaded. Ensure model_final.h5 exists."
        }
        
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize to 224x224 as expected by the model
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        x = np.array(image, dtype=np.float32)
        x = x / 255.0  # Normalization step used in training
        x = np.expand_dims(x, axis=0) # Add batch dimension (1, 224, 224, 3)
        
        # Predict
        predictions = model.predict(x)
        probabilities = predictions[0]
        predicted_idx = np.argmax(probabilities)
        
        predicted_class = CLASSES[predicted_idx]
        confidence_score = float(probabilities[predicted_idx]) * 100
        
        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": f"{confidence_score:.2f}%",
            "class_index": int(predicted_idx),
            "all_probabilities": {CLASSES[i]: f"{float(probabilities[i])*100:.2f}%" for i in range(len(CLASSES))}
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
