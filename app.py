import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
from PIL import Image

# -----------------------------
# Load Model & Class Indices
# -----------------------------
MODEL_PATH = "models/final_model.h5"   # Use your saved model path
CLASS_INDICES_PATH = "models/class_indices.json"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Detect input size automatically from model
input_shape = model.input_shape  # e.g., (None, 150, 150, 3)
IMG_HEIGHT, IMG_WIDTH = input_shape[1], input_shape[2]

# Load class indices mapping
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping safely (handles both NORMAL/PNEUMONIA orderings)
idx_to_class = {v: k for k, v in class_indices.items()}

# -----------------------------
# Helper: Preprocess Image
# -----------------------------
def preprocess_image(img):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize for model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 Pneumonia Detection App")
st.write("Upload a chest X-ray image to check if it's **Normal** or shows signs of **Pneumonia**.")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded X-ray")

    # Preprocess
    img = Image.open(uploaded_file).convert("RGB")
    processed_img = preprocess_image(img)

    # Predict
    prediction = model.predict(processed_img)

    # -----------------------------
    # Handle Sigmoid vs Softmax
    # -----------------------------
    if prediction.shape[1] == 1:  # Sigmoid binary classifier
        score = prediction[0][0]
        pred_idx = 1 if score >= 0.5 else 0
        confidence = score if pred_idx == 1 else 1 - score
    else:  # Softmax multi-class (2 classes)
        pred_idx = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][pred_idx]

    # Map predicted index to class name
    pred_class = idx_to_class.get(pred_idx, f"Class {pred_idx}")

    # Show result
    st.subheader(f"🔍 Prediction: **{pred_class}**")
    st.write(f"Confidence Score: {confidence:.4f}")
