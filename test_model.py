import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

def test_model():
    print("Loading model...")
    model = load_model('models/final_model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\nModel input shape:", model.input_shape)
    print("\nModel summary:")
    model.summary()
    
    # Test on a sample image from your test directory
    test_dir = "test"
    if os.path.exists(test_dir):
        # Find first image in NORMAL and PNEUMONIA folders
        normal_img = None
        pneumonia_img = None
        
        normal_dir = os.path.join(test_dir, "NORMAL")
        pneumonia_dir = os.path.join(test_dir, "PNEUMONIA")
        
        if os.path.exists(normal_dir):
            normal_files = os.listdir(normal_dir)
            if normal_files:
                normal_img = os.path.join(normal_dir, normal_files[0])
        
        if os.path.exists(pneumonia_dir):
            pneumonia_files = os.listdir(pneumonia_dir)
            if pneumonia_files:
                pneumonia_img = os.path.join(pneumonia_dir, pneumonia_files[0])
        
        def predict_image(image_path):
            if image_path is None:
                return
            
            print(f"\nTesting image: {image_path}")
            img = load_img(image_path, target_size=(224, 224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            pred = model.predict(x)
            print(f"Raw prediction value: {pred[0][0]}")
            print(f"Predicted class: {'PNEUMONIA' if pred[0][0] > 0.5 else 'NORMAL'}")
            print(f"Confidence: {max(pred[0][0], 1 - pred[0][0]) * 100:.2f}%")
        
        if normal_img:
            predict_image(normal_img)
        if pneumonia_img:
            predict_image(pneumonia_img)
    else:
        print("Test directory not found!")

if __name__ == "__main__":
    test_model()