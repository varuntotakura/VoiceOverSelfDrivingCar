###
# Copyright (2019). All Rights belongs to VARUN
# Use the code by mentioning the Sredits
# Credit: github.com/t-varun
# Developer:
#
#               T VARUN
#
###

# Import the required libraries
import numpy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time

# Import th data
data_name = 'training_data_cleaned.npy'
data = np.load(data_name)

# Declare the required arrays
img = []
label = []
test_imgs = []
test_labs = []

# Class names 
class_names = ['Angry_1', 'Angry_2', 'Angry_3', 'Angry_4', 'Angry_5', 
        'BAS_1', 'BAS_2', 'BAS_3','BAS_4', 'BAS_5', 'download (1)',
        'download', 'Hungry_1', 'Hungry_2', 'Owner_1']

# Input to the arrays
for item, index in data:
    img.append(item)
    label.append(index)
    if class_names[index] == 'download' or class_names[index] == 'download (1)':
        test_imgs.append(item)
        test_labs.append(index)

# Train and Test data
train_images = img[-50:]
train_labels = label[-50:]

test_images = img[:-5]
test_labels = label[:-5]

train_images = np.asarray(train_images)
test_images = np.asarray(test_images)
test_imgs = np.asarray(test_imgs)

train_images = train_images.reshape((-1, 60, 80, 1))
test_images = test_images.reshape((-1, 60, 80, 1))
test_imgs = test_imgs.reshape((-1, 60, 80, 1))

# Image Processing
train_images = train_images / 255.0

test_images = test_images / 255.0

test_imgs = test_imgs / 255.0

# Sequential Model
# Convolutional Neural Network
model = keras.Sequential([
    keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(60, 80, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
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
model.fit(train_images, train_labels, epochs=3)

model.save('SelfDrivingCar-TensorModel.model')

# Print the Summary
model.summary()

# Accuracy of the Model
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Print Test accuracy
print('Test accuracy:', test_acc)

# Make Predictions
predictions = model.predict([test_imgs])[0] 
predicted_label = class_names[np.argmax(predictions)]

# Compare the predictions
print("Predictions : ",predicted_label)
print("Actual : ",class_names[test_labs[0]])

##print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('Model')
plt.ylabel('Result')
plt.xlabel('Epochs')
plt.legend(['Accuracy', 'Loss'], loc='upper right')
plt.show()
