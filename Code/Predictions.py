from PIL import ImageGrab
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from Directkeys import PressKey, ReleaseKey, W, S, A, D
import win32api as wapi
import random
import time

Name = 'Self-Driving-Convelutional-Network'

class_names = ['W', 'S', 'A', 'D']

model = tf.keras.models.load_model(Name + '.model')
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

def right():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    
def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

def keys_to_output(keys):
    #         w s a d
    output = [0,0,0,0]    
    if 'W' in keys:
        output = [1,0,0,0]
    elif 'S' in keys:
        output = [0,1,0,0]
    elif 'A' in keys:
        output = [0,0,1,0]
    elif 'D' in keys:
        output = [0,0,0,1]    
    return output

def record():
    Capturing = []
    while True:
        capture_img = np.array(ImageGrab.grab())
        capture_img = cv2.resize(capture_img, (150,100))
        capture_img = cv2.cvtColor(capture_img, cv2.COLOR_BGR2RGB)
        keys = key_check()
        if not 'T' in keys:
            output = keys_to_output(keys)
            Capturing.append([capture_img, output])
        else:
            return Capturing

filename = 'Data_Captured.npy'
try:
    Capture = list(np.load(filename))
except:
    Capture = []
    
paused = False

while True:
    if not paused:
        # printscreen_pil = np.array(ImageGrab.grab(bbox=(10,10,860,670)))
        printscreen_pil = np.array(ImageGrab.grab())
        processed_img = cv2.cvtColor(printscreen_pil, cv2.COLOR_BGR2RGB)
        processed_img = cv2.resize(processed_img, (150,100))
        processed_img = processed_img.reshape((-1, 100, 150, 3))
        predictions = model.predict([processed_img])[0]
        predicted_label = list(np.around(predictions))
        print("Moves: ", predicted_label, "Predictions: ", predictions)
        
        if predicted_label == [1.0, 0.0, 0.0, 0.0]:
            straight()
        elif predicted_label == [0.0, 1.0, 0.0, 0.0]:
            reverse()
        elif predicted_label == [0.0, 0.0, 1.0, 0.0]:
            left()
        elif predicted_label == [0.0, 0.0, 0.0, 1.0]:
            right()
        else:
            reverse()

    keys = key_check()
    if 'T' in keys:
        if paused:
            paused = False
            time.sleep(1)
            Capture.append(record())
        else:
            paused = True
            ReleaseKey(A)
            ReleaseKey(D)
            ReleaseKey(W)
            ReleaseKey(S)
            time.sleep(1)
    if 'Q' in keys or cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()
np.save(filename, Capture)

####print(history.history.keys())
##
### summarize history for accuracy
##plt.plot(history.history['acc'])
##plt.plot(epochs)
##plt.title('Model')
##plt.ylabel('Result')
##plt.xlabel('Epochs')
##plt.legend(['Accuracy', 'Loss'], loc='upper right')
##plt.show()
