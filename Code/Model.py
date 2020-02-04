###
# Copyright (2020). All Rights belongs to VARUN
# Use the code by mentioning the Credits
# Developer:
#
#               T VARUN
#
###

# Import the required libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import os
import matplotlib.pyplot as plt

# Import th data
Data = '../Data/data.npy'
data = np.load(Data, allow_pickle=True)
Name = '../Code/TrainedModel/Voice-Over-Self-Driving-Convolutional-Network'
tensorboard = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(Name))

# Declare the required arrays
imgs = []
labels = []

# Class names 
class_names = ['W', 'S', 'A', 'D']

# Input to the arrays
for img, keys in data:
    imgs.append(img)
    if keys == [1, 0, 0, 0]:
        label = 0
    elif keys == [0, 1, 0, 0]:
        label = 1
    elif keys == [0, 0, 1, 0]:
        label = 2
    elif keys == [0, 0, 0, 1]:
        label = 3
    else:
        label = 0
    labels.append(label)

##print(len(imgs), len(labels))

# Train and Test data
train_images = imgs[:-8]
train_labels = labels[:-8]

test_images = imgs[-8:]
test_labels = labels[-8:]

train_images = np.asarray(train_images)
test_images = np.asarray(test_images)

train_images = train_images.reshape((-1, 60, 80, 1))
test_images = test_images.reshape((-1, 60, 80, 1))

# Image Processing
train_images = train_images / 255.0
test_images = test_images / 255.0

# Sequential Model
# Convolutional Neural Network
model = keras.Sequential([
    keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(60, 80, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(train_images, train_labels, epochs=100, callbacks=[tensorboard])
## tensorboard --logdir=logs/ --host=127.0.0.1

# Save the Model
model.save(Name + '.model')

# Print the Summary
model.summary()

# Accuracy of the Model
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Print Test accuracy
print('Test accuracy:', test_acc*100, '%')

# Make Predictions
predictions = model.predict([test_images])[0] 
predicted_label = class_names[np.argmax(predictions)]

# Compare the predictions
print("Predictions : ",predicted_label)
print("Actual : ",class_names[test_labels[0]])

##print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('Model')
plt.ylabel('Result')
plt.xlabel('Epochs')
plt.legend(['Accuracy', 'Loss'], loc='upper right')
plt.show()
