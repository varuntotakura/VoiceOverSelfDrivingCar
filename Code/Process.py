import numpy as np
import cv2
import time
import win32api as wapi
import os
import sys
from PIL import Image
from random import shuffle
import pandas as pd

file_name = '../Data/train_data.npy'

if os.path.isfile(file_name):
    print("File exists, loading previous data")
    training_data = list(np.load(file_name))
else:
    print("File does not exist, stating fresh")
    training_data = []

last_time = time.time()

path_train_main = '../Images/Stop/'
path_train = [os.path.join(path_train_main, f) for f in os.listdir(path_train_main)]
print(path_train)

##output = [1,0,0,0]W
##output = [0,1,0,0]S
##output = [0,0,1,0]A
##output = [0,0,0,1]D

output = [0,1,0,0]
for file in path_train:
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img',img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (80,60))
    training_data.append([img, output])
    last_time = time.time()
    if cv2.waitKey(30) & 0xff == 'q' == 27:
        break
    np.save(file_name, training_data)
    cv2.destroyAllWindows()
