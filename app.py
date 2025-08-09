from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import json
import os
import base64
from PIL import Image
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

class HandGestureAPI:
    def __init__(self, model_path='models/best_gesture_model.h5', 
                 class_names_path='models/class_names.json'):
        self.img_height = 128
        self.img_width = 128
        self.model = None
        self.class_names = []
        
        # Gesture mapping from folder names to readable names
        self.gesture_mapping = {
            '01_palm': 'Palm',
            '02_l': 'PALM',
            '03_fist': 'Fist',
            '04_fist_moved': 'Fist (Moved)',
            '05_thumb': 'Thumbs Up',
            '06_index': 'Index Finger',
            '07_ok': 'OK Sign',
            '08_palm_moved': 'Palm (Moved)',
            '09_c': 'C Shape',
            '10_down': 'Thumbs Down'
        }
        
        # Load model and class names
        self.load_model(model_path, class_names_path)
    
    def load_model(self, model_path, class_names_path):
        """Load trained model and class names"""
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.error(f"Model file not found: {model_path}")
                return False
                
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
                logger.info(f"Class names loaded: {self.class_names}")
            else:
                # Fallback to default class names
                self.class_names = [
                    '01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
                    '06_index', '07_ok', '08_palm_moved', '09_c', '10_down'
                ]
                logger.warning("Using default class names")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        try:
            # Resize image
            image = cv2.resize(image, (self.img_width, self.img_height))
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def predict_gesture(self, image):
        """Predict gesture from image"""
        try:
            if self.model is None:
                return None, None, "Model not loaded"
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None, None, "Image preprocessing failed"
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get technical class name
            if predicted_class_idx < len(self.class_names):
                technical_name = self.class_names[predicted_class_idx]
            else:
                technical_name = f"Class_{predicted_class_idx}"
            
            # Convert to readable gesture name
            readable_name = self.gesture_mapping.get(technical_name, technical_name)
            
            return readable_name, confidence, None
            
        except Exception as e:
            logger.error(f"Error predicting gesture: {str(e)}")
            return None, None, str(e)

# Initialize the gesture recognizer
gesture_api = HandGestureAPI()

@app.route('/')
def index():
    """Serve the main page"""
    return jsonify({
        "message": "Hand Gesture Recognition API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Predict gesture from base64 image",
            "/health": "GET - Health check",
            "/classes": "GET - Get available gesture classes"
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_loaded = gesture_api.model is not None
    return jsonify({
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "classes_count": len(gesture_api.class_names)
    })

@app.route('/classes')
def get_classes():
    """Get available gesture classes"""
    # Return both technical and readable names
    classes_info = []
    for technical_name in gesture_api.class_names:
        readable_name = gesture_api.gesture_mapping.get(technical_name, technical_name)
        classes_info.append({
            "technical_name": technical_name,
            "readable_name": readable_name
        })
    
    return jsonify({
        "classes": classes_info,
        "count": len(gesture_api.class_names),
        "readable_names_only": [info["readable_name"] for info in classes_info]
    })

@app.route('/predict', methods=['POST'])
def predict_gesture():
    """Predict gesture from uploaded image"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode base64 image
        try:
            image_data = data['image']
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            return jsonify({"error": f"Invalid image format: {str(e)}"}), 400
        
        # Make prediction
        gesture_name, confidence, error = gesture_api.predict_gesture(image_cv)
        
        if error:
            return jsonify({"error": error}), 500
        
        return jsonify({
            "gesture": gesture_name,
            "confidence": confidence,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict gestures from multiple images"""
    try:
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({"error": "No images data provided"}), 400
        
        results = []
        
        for idx, image_data in enumerate(data['images']):
            try:
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                
                # Convert to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Make prediction
                gesture_name, confidence, error = gesture_api.predict_gesture(image_cv)
                
                if error:
                    results.append({
                        "index": idx,
                        "error": error,
                        "success": False
                    })
                else:
                    results.append({
                        "index": idx,
                        "gesture": gesture_name,
                        "confidence": confidence,
                        "success": True
                    })
                    
            except Exception as e:
                results.append({
                    "index": idx,
                    "error": f"Invalid image format: {str(e)}",
                    "success": False
                })
        
        return jsonify({
            "results": results,
            "total_processed": len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in predict_batch endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Check if model files exist
    if not os.path.exists('models/'):
        logger.warning("Models directory not found. Please ensure you have trained the model first.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)