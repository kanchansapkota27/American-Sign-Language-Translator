import cv2
import numpy as np
import os

path="./dataset/"
gestures = os.listdir(path)
print(gestures)
x, y = [], []

dict_labels = {
        'A': 1,
        'B': 2,
        'C':3,
        'D':4,
        'E':5,
        'F':6,
        'G':7,
        'H':8,
        'I':9,
        'K':10,
        'L':11,
        'M':12,
        'BG':13,
        'DEL':14,
        'SPACE':15,
    }

for i in gestures:
    print(i)
    images = os.listdir(path + i)
    for j in images:
        img_path = path + i + '/' + j
        img = cv2.imread(img_path,0)
        img = img.reshape((50,50,1))
        img = img/255.0
        x.append(img)
        y.append(dict_labels[i])

x = np.array(x)
y = np.array(y)

np.save("X_array",x)
np.save("Y_array",y)
