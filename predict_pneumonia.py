import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺")
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload one or more chest X-ray images to predict NORMAL or PNEUMONIA.")

# -----------------------------
# Load SavedModel
# -----------------------------
MODEL_PATH = "models/final_model.h5"

try:
    model = load_model(MODEL_PATH, compile=False)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Get input size automatically
# -----------------------------
input_height, input_width = model.input_shape[1], model.input_shape[2]

# -----------------------------
# File uploader
# -----------------------------
uploaded_files = st.file_uploader(
    "Choose X-ray images", 
    type=["jpg","jpeg","png"], 
    accept_multiple_files=True
)

# -----------------------------
# Preprocess images
# -----------------------------
def preprocess_image(img, target_size=(input_width, input_height)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# -----------------------------
# Make predictions
# -----------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        # Preprocess and predict
        img_array = preprocess_image(img, target_size=(input_width, input_height))
        prediction = model.predict(img_array)

        # Binary or softmax
        if prediction.shape[1] == 1:  # sigmoid output
            pred_class = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
        else:  # softmax output
            pred_class = "NORMAL" if np.argmax(prediction[0]) == 0 else "PNEUMONIA"

        st.write(f"**Prediction:** {pred_class}")
