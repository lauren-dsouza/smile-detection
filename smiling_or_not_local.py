"""
Smiling or Not Classifier
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import kagglehub

# Download latest version
DATASET_PATH = kagglehub.dataset_download("chazzer/smiling-or-not-face-data")

print("Path to dataset files:", DATASET_PATH)

# Set this to your local dataset path (should contain subfolders for each class)
# DATASET_PATH = "./data/smiling-or-not"

BATCH_SIZE = 32
IMG_SIZE = (96, 96)

# Load dataset from local directory
train_val_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',
    label_mode='binary',
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# Split dataset into train and validation sets

dataset_size = len(train_val_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset = train_val_dataset.take(train_size)
val_dataset = train_val_dataset.skip(train_size)

# Visualize data samples
class_names = train_val_dataset.class_names
print("Class names:", class_names)
plt.figure(figsize=(10, 10))
for images, labels in train_val_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        idx = int(labels[i].numpy()[0]) if labels.shape[-1] == 1 else int(labels[i].numpy())
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[idx])
        plt.axis("off")
plt.show()

# Model creation
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet',
    alpha=0.35
)

# Freeze the convolutional base
base_model.trainable = False

# Preprocessing
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Classification head
classification_input_layer = tf.keras.layers.Flatten()
prediction_layer = tf.keras.layers.Dense(1)

def build_model():
    inputs = tf.keras.Input(shape=(96, 96, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = classification_input_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model

model = build_model()

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# Training and evaluation
EPOCHS = 5
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# Plot training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(val_dataset)
print('Test accuracy :', accuracy)

# Visualize predictions on validation set
image_batch, label_batch = next(iter(val_dataset))
predictions = model.predict_on_batch(image_batch).flatten()
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch.numpy().reshape(-1).astype("uint8"))

plt.figure(figsize=(10, 18))
for i in range(min(18, len(image_batch))):
    ax = plt.subplot(6, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    plt.title(class_names[int(predictions[i])])
    plt.axis("off")
plt.show()

# Predict on your own images (place images in a folder, e.g., './predict_images')
from tensorflow.keras.utils import load_img, img_to_array
PREDICT_FOLDER = "./predict_images"  # <-- CHANGE THIS to your folder with images to predict
if os.path.isdir(PREDICT_FOLDER):
    for fn in os.listdir(PREDICT_FOLDER):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(PREDICT_FOLDER, fn)
            img = load_img(path, target_size=(96, 96))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            classes = model.predict(x)
            print(f"{fn}: {classes[0][0]:.3f} -> {'not a smile' if classes[0][0]>0.5 else 'smile'}")
else:
    print(f"Prediction folder '{PREDICT_FOLDER}' not found. Skipping custom image prediction.")
