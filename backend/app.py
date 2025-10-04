from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes with specific configuration

# Load the model and get input dimensions
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'final_model.h5')
model = load_model(MODEL_PATH)
IMG_HEIGHT, IMG_WIDTH = model.input_shape[1:3]

def preprocess_image(image_bytes):
    """Preprocess the uploaded image for model prediction"""
    # Open image from bytes
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize image to match model's expected input shape
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to numpy array and normalize
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess the image
        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Process prediction (assuming binary classification with sigmoid)
        probability = float(prediction[0][0])
        is_pneumonia = probability >= 0.5
        
        result = {
            'prediction': 'Pneumonia' if is_pneumonia else 'Normal',
            'confidence': probability if is_pneumonia else 1 - probability
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)