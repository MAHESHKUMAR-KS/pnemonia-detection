# convert_model.py
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model

# -----------------------------
# Paths
# -----------------------------
H5_MODEL_PATH = "D:/trained-models/final_model.h5"         # Original model with batch_shape issue
WEIGHTS_PATH = "D:/trained-models/final_model_weights.h5"  # Optional: weights-only
SAVED_MODEL_DIR = "D:/trained-models/final_model_savedmodel"

# -----------------------------
# Rebuild the model architecture
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Change to Dense(2, activation='softmax') if original was softmax
])

# -----------------------------
# Load weights safely
# -----------------------------
try:
    if os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH, by_name=True)
        print("✅ Weights loaded successfully.")
    elif os.path.exists(H5_MODEL_PATH):
        # If only H5 exists, try loading via load_model with compile=False
        tmp_model = load_model(H5_MODEL_PATH, compile=False)
        model.set_weights(tmp_model.get_weights())
        print("✅ H5 model weights loaded successfully.")
    else:
        print("⚠ No weights or H5 model found. Model is untrained.")
except Exception as e:
    print(f"❌ Error loading weights: {e}")

# -----------------------------
# Save as SavedModel
# -----------------------------
try:
    model.save(SAVED_MODEL_DIR)
    print(f"✅ Model saved successfully to '{SAVED_MODEL_DIR}'")
except Exception as e:
    print(f"❌ Error saving model: {e}")
