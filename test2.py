# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 20:08:37 2020

@author: Admin
"""
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
import pickle
from keras.models import model_from_json
from scipy import sparse
import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import model_from_json
from scipy import sparse

EPOCHS = 25
INIT_LR = 1e-3
BS = 64
default_image_size = tuple((256, 256))
image_size = 0
directory_root = 'corn1/'
width=256
height=256
depth=3



def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
    


EPOCHS = 25
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
lb = pickle.load(open("label_transform.pkl","rb"))

INIT_LR = 1e-3
opt=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
loaded_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('im1.jpg')
img = cv2.resize(img,(256,256))
img = np.reshape(img,[1,256,256,3])
img=convert_image_to_array(img)
np_image_li = np.array(img, dtype=np.float16) / 225.0
img = np.expand_dims(np_image_li, 0)


classes = loaded_model.predict_classes(img)



sA = sparse.csr_matrix(classes)
lb = pickle.load(open("label_transform.pkl","rb"))
disease = f"{lb.inverse_transform(sA)[0]}"
print("The Disease detected is :",disease)

img2 = cv2.imread('im1.jpg')
plt.imshow(img2,)