# deep-learning

COMPANY : CODTECH IT SOLUTIONS

NAME : MIRIYALA SUVARNA

INTERN ID : CITS13

DOMAIN : DATA SCIENCE

DURATION : 6WEEKS

MENTOR : NEELA SANTOSH KUMAR
DESCRIPTION :
# Deep Learning Image Classification using TensorFlow

## Overview

This project implements an image classification model using Deep Learning techniques with TensorFlow and Keras. A Convolutional Neural Network (CNN) was developed to classify images from the CIFAR-10 dataset into ten different categories, including airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Objectives

* Build a Deep Learning model for image classification.
* Perform image preprocessing and normalization.
* Train and evaluate a Convolutional Neural Network (CNN).
* Visualize model performance using accuracy graphs.
* Generate predictions on unseen test images.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Google Colab

## Dataset

The project uses the CIFAR-10 dataset, which contains 60,000 color images of size 32×32 pixels. The dataset is divided into 50,000 training images and 10,000 testing images across 10 classes.

## Methodology

1. Loaded and explored the CIFAR-10 dataset.
2. Normalized image pixel values to improve model training.
3. Built a CNN architecture consisting of:

   * Conv2D Layer
   * MaxPooling2D Layer
   * Flatten Layer
   * Dense Layers
4. Compiled the model using the Adam optimizer and Sparse Categorical Crossentropy loss function.
5. Trained the model for 5 epochs.
6. Evaluated model performance on the test dataset.
7. Visualized training and validation accuracy.

## Results

The trained CNN model achieved approximately 62% test accuracy on the CIFAR-10 dataset. Training and validation accuracy graphs were generated to analyze model performance, and sample predictions were tested on unseen images.

## Conclusion

This project demonstrates the implementation of a Deep Learning-based image classification system using TensorFlow. The CNN successfully learned image features and classified images into multiple categories, providing practical experience with Deep Learning workflows and image recognition tasks.

CODE :

#import all the required libraries
"""tensorflow → Deep learning framework.
datasets → Provides ready-made datasets like CIFAR-10.
layers → Used to build neural network layers.
models → Used to create the complete neural network.
matplotlib → Used for graphs and image visualization.
numpy → Used for numerical operations."""
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)

print("Testing images shape:", test_images.shape)
print("Testing labels shape:", test_labels.shape)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(8,8))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(train_images[i])
    plt.title(class_names[train_labels[i][0]])
    plt.axis('off')

plt.show()
train_images = train_images / 255.0
test_images = test_images / 255.0
print(train_images[0].min())
print(train_images[0].max())
model = models.Sequential()

model.add(layers.Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(32,32,3)
))

model.add(layers.MaxPooling2D((2,2)))

model.summary()
model = models.Sequential([
    layers.Input(shape=(32,32,3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels)
)
test_loss, test_acc = model.evaluate(
    test_images,
    test_labels,
    verbose=2
)

print("Test Accuracy:", test_acc)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.show()

OUTPUT  :

<img width="1096" height="765" alt="Image" src="https://github.com/user-attachments/assets/1a7e17fd-b082-403a-81c3-aa4fd80f8650" />
<img width="1096" height="765" alt="Image" src="https://github.com/user-attachments/assets/1a7e17fd-b082-403a-81c3-aa4fd80f8650" />
