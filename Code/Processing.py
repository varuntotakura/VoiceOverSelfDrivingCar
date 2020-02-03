import numpy as np
import pandas as pd
import cv2
from random import shuffle

train_data = np.load('../Data/train_data.npy')

TOTAL = []

for data in train_data:
    img = data[0]
    keys = data[1]
    TOTAL.append([img, keys])
    
shuffle(TOTAL)
np.save('../Data/data.npy', TOTAL)

t_data = np.load('../Data/data.npy')
for dat in t_data:
    key = dat[1]
    print(key)
