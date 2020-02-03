import numpy as np
import cv2
import time
import win32api as wapi
import os
import sys
from PIL import Image
from random import shuffle
import pandas as pd

file_name_1 = 'train_data.npy'
file_name_2 = 'test_data.npy'

if os.path.isfile(file_name_1):
    print("File exists, loading previous data")
    training_data = list(np.load(file_name_1))
else:
    print("File does not exist, starting fresh")
    training_data = []

if os.path.isfile(file_name_2):
    print("File exists, loading previous data")
    testing_data = list(np.load(file_name_2))
else:
    print("File does not exist, starting fresh")
    testing_data = []

last_time = time.time()

path_train_main = './FinalImages/Train/'
classes_train = [os.path.join(path_train_main, f).split('/')[-1] for f in os.listdir(path_train_main)]
# print(classes_train)

for cl in classes_train:
    path = path_train_main+cl+'/'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for index, file in enumerate(imagePaths):
        name = 'FinalImages/Train/'+cl+'/'+file.split('/')[-1]
        try:
            img = cv2.imread(name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('img',img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (80,60))
            training_data.append([img, cl])
            # print(training_data)
            # print('Frame took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            if cv2.waitKey(30) & 0xff == 'q' == 27:
                break
        except:
            np.save(file_name_1, training_data)        
    np.save(file_name_1, training_data)
    cv2.destroyAllWindows()

path_test_main = './FinalImages/Test/'
classes_test = [os.path.join(path_test_main, f).split('/')[-1] for f in os.listdir(path_test_main)]
# print(classes_test)

for cl in classes_test:
    path = path_test_main+cl+'/'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for index, file in enumerate(imagePaths):
        name = 'FinalImages/Test/'+cl+'/'+file.split('/')[-1]
        try:
            img = cv2.imread(name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('img',img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (80,60))
            testing_data.append([img, cl])
            # print(testing_data)
            # print('Frame took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            if cv2.waitKey(30) & 0xff == 'q' == 27:
                break
        except:
            np.save(file_name_2, testing_data)        
    np.save(file_name_2, testing_data)
    cv2.destroyAllWindows()

train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

TOTAL_TRAIN = []
TOTAL_TEST = []
classes = np.unique(classes_train + classes_test)
# print(classes)

for index, data in enumerate(train_data):
    name = data[1]
    # print(np.shape(data[0]))
    for i in range(len(classes)):
        if name == classes[i]:
            TOTAL_TRAIN.append([data[0], i])

for index, data in enumerate(test_data):
    name = data[1]
    # print(np.shape(data[0]))
    for i in range(len(classes)):
        if name == classes[i]:
            TOTAL_TEST.append([data[0], i])

shuffle(TOTAL_TRAIN)
shuffle(TOTAL_TEST)
np.save('train_data_cleaned.npy', TOTAL_TRAIN)
np.save('test_data_cleaned.npy', TOTAL_TEST)
