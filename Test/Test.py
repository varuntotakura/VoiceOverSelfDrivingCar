'''
## Licence:

This repository contains a variety of content; some developed by VARUN, and some from third-parties.
The third-party content is distributed under the license provided by those parties.
The content developed by VARUN is distributed under the following license:
I am providing code and resources in this repository to you under an open source license.
Because this is my personal repository, the license you receive to my code and resources is from me.

More about Licence at [link](https://github.com/t-varun/Face-Recognition/blob/master/LICENSE).
'''

# Import the requirements
import cv2
import numpy as np
import tensorflow as tf

CATEGORIES = ['VARUN', 'BUNNY']

data_name = '../training_data_cleaned.npy'

data = np.load(data_name)

img = []

for item in data:
    img.append(item[0])

test_images = img[:500]

test_images = np.asarray(test_images)
test_images = test_images / 255.0

def prepare(path):
    img_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    res_arr = cv2.resize(img_arr, (80, 60))
    res = np.argmax(res_arr)
    res = res/255.0
    return res

model = tf.keras.models.load_model('FR-TensorModel.model')

Input = prepare('_DSC0380.jpg')

prediction = model.predict(test_images)
predicted_label = np.argmax(prediction)
print(predicted_label)
