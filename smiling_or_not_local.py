"""
Smiling or Not Classifier
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
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

# Model creation
IMG_SHAPE = IMG_SIZE + (3,)
base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet',
    alpha=0.35
)

# Freeze the convolutional base
base_model.trainable = False

# Preprocessing
preprocess_input = keras.applications.mobilenet_v2.preprocess_input

# Classification head
classification_input_layer = keras.layers.Flatten()
prediction_layer = keras.layers.Dense(1)

def build_model():
    inputs = keras.Input(shape=(96, 96, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = classification_input_layer(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = keras.Model(inputs, outputs)
    return model

model = build_model()

base_learning_rate = 0.0001
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# Training and evaluation
EPOCHS = 10
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

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
new_head = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1)
])

# Attach the new head to the frozen model
inputs = model.input
x = model.output
x = new_head(x)
transfer_model = keras.Model(inputs, x)

# Compile the new model
transfer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
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

'''

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
      )

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/content/train-horse-or-human/',  # This is the source directory for training images
        target_size=(96,96),  # All images will be resized to 100x100
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')


# VALIDATION
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        '/content/validation-horse-or-human',
        target_size=IMAGE_SIZE,
        class_mode='binary')
'''


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
