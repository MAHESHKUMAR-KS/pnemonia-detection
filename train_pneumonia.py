import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = r"D:\pneumonia_project"   # <-- updated
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")
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
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

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
# Build Model with VGG16 base
# -----------------------------
# Load VGG16 with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight
classes = np.array([0, 1])  # Binary classification
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_generator.classes
)
class_weight_dict = dict(zip(classes, class_weights))

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy", 
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()]
)

# -----------------------------
# Train
# -----------------------------
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Train with more epochs and all callbacks
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    class_weight=class_weight_dict,
    callbacks=[
        early_stopping,
        model_checkpoint,
        lr_scheduler,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
)

# Print final metrics
final_metrics = model.evaluate(val_generator)
metric_names = ['Loss', 'Accuracy', 'AUC', 'Precision', 'Recall']
print("\n=== Final Validation Metrics ===")
for name, value in zip(metric_names, final_metrics):
    print(f"{name}: {value:.4f}")

# Print final metrics
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"\n✅ Final Training Accuracy: {final_train_accuracy:.2%}")
print(f"✅ Final Validation Accuracy: {final_val_accuracy:.2%}")

# -----------------------------
# Save Model
# -----------------------------
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
