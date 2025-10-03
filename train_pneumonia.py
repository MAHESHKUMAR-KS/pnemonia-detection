import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = r"D:\pneumonia_project"   # <-- updated
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

# -----------------------------
# Data Preprocessing
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

# -----------------------------
# Save Class Indices
# -----------------------------
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(train_generator.class_indices, f)

print("✅ Class indices saved:", train_generator.class_indices)

# -----------------------------
# Build Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# -----------------------------
# Save Model
# -----------------------------
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
