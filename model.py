import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import cifar10
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()

# Normalize the data
training_images = training_images / 255.0
testing_images = testing_images / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
datagen.fit(training_images)

# Define the more complex CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    GlobalAveragePooling2D(),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(10, activation='softmax')
])

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Implement learning rate scheduling and early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# Train the model with data augmentation
history = model.fit(datagen.flow(training_images, training_labels, batch_size=64),
                    epochs=50,
                    validation_data=(testing_images, testing_labels),
                    callbacks=[early_stopping, lr_scheduler])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(testing_images, testing_labels)
print(f"Test accuracy: {test_accuracy:.4f}")

# Predict and evaluate model predictions
predictions = model.predict(testing_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_labels == testing_labels.flatten())
print(f"Model accuracy on test set: {accuracy:.4f}")

# Run this code to create a dictionary that translates the labels to meaningful strings
class_labels = {
    0 : "airplane",
    1 : "automobile",
    2 : "bird",
    3 : "cat",
    4 : "deer",
    5 : "dog",
    6 : "frog",
    7 : "horse",
    8 : "ship",
    9 : "truck"
}

# Misclassified images
misclassified_indices = np.where(predicted_labels != testing_labels.flatten())[0]

# Display a few misclassified images
plt.figure(figsize=(10, 10))
for i, idx in enumerate(misclassified_indices[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(testing_images[idx])
    plt.title(f"True: {class_labels[testing_labels[idx][0]]}, Pred: {class_labels[predicted_labels[idx]]}")
    plt.axis('off')
plt.suptitle("Misclassified Images", fontsize=16)
plt.show()

# Plot the training, validation, and test accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

# Plot the test accuracy as a horizontal line
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training, Validation, and Test Accuracy')
plt.legend()
plt.show()
