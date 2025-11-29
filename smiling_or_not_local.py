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

# Load dataset from local directory, ignoring the 'test' folder
# We'll use only 'smile' and 'non_smile' subfolders for training/validation

# List only 'smile' and 'non_smile' directories
allowed_classes = ['smile', 'non_smile']

train_val_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',
    label_mode='binary',
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    class_names=allowed_classes
)

# Split dataset into train and validation sets

dataset_size = len(train_val_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset = train_val_dataset.take(train_size)
val_dataset = train_val_dataset.skip(train_size)

# Visualize data samples
# class_names = train_val_dataset.class_names
# print("Class names:", class_names)
# plt.figure(figsize=(10, 10))
# for images, labels in train_val_dataset.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         idx = int(labels[i].numpy()[0]) if labels.shape[-1] == 1 else int(labels[i].numpy())
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[idx])
#         plt.axis("off")
# plt.show()

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
EPOCHS = 10
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# Plot training and validation loss
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# plt.figure()
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0,1.0])
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()

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

# Define class_names for plotting
class_names = allowed_classes

plt.figure(figsize=(10, 18))
for i in range(min(18, len(image_batch))):
    ax = plt.subplot(6, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    plt.title(class_names[int(predictions[i])])
    plt.axis("off")
plt.show()

# Freeze all layers in the current model
for layer in model.layers:
    layer.trainable = False

# Add a new trainable classification head for transfer learning
new_head = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
])

# Attach the new head to the frozen model
inputs = model.input
x = model.output
x = new_head(x)
transfer_model = tf.keras.Model(inputs, x)

# Compile the new model
transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Now you can train transfer_model on your own imported data
DATASET_PATH = "./data"
train_val_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',
    label_mode='binary',
    shuffle=True,
    batch_size=1,
    image_size=IMG_SIZE
)

dataset_size = len(train_val_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
print("Train size:", train_size)
print("Validation size:", val_size)
print("Dataset size:", dataset_size)
train_dataset = train_val_dataset.take(train_size)
val_dataset = train_val_dataset.skip(train_size)

EPOCHS = 10
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

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
