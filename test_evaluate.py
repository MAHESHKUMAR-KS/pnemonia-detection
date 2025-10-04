import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Correct test directory path
TEST_DIR = "test"

# Image size must match training
IMG_SIZE = (150, 150)

# Data generator (no augmentation for test set)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Load model
model = tf.keras.models.load_model("D:/trained-models/final_model_savedmodel")

# Compile (needed for evaluation)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Evaluate
loss, acc = model.evaluate(test_generator, verbose=1)
print(f"✅ Test Accuracy: {acc*100:.2f}%")
print(f"✅ Test Loss: {loss:.4f}")
