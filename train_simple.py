import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model_new.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

# -----------------------------
# Data Preprocessing
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255.0
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),  # Smaller input size for faster training
    batch_size=32,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(150, 150),  # Smaller input size for faster training
    batch_size=32,
    class_mode="binary"
)

# -----------------------------
# Save Class Indices
# -----------------------------
os.makedirs(os.path.dirname(CLASS_INDICES_PATH), exist_ok=True)
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(train_generator.class_indices, f)

print("✅ Class indices saved:", train_generator.class_indices)

# -----------------------------
# Build Model
# -----------------------------
model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Dense Layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.AUC(),
             tf.keras.metrics.Precision(), 
             tf.keras.metrics.Recall()]
)

# Calculate class weights
weight_for_0 = (1 / len(train_generator.classes[train_generator.classes == 0])) * (len(train_generator.classes) / 2.0)
weight_for_1 = (1 / len(train_generator.classes[train_generator.classes == 1])) * (len(train_generator.classes) / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print("\nModel Summary:")
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    class_weight=class_weight,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )
    ]
)

# Save the model
print("\nSaving the model...")
model.save(MODEL_PATH)
print("✅ Model saved to:", MODEL_PATH)