# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 21:39:38 2020

@author: Admin
"""


from keras.models import load_model


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
from sklearn.metrics import classification_report,confusion_matrix






CATEGORIES = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'
  'Corn_(maize)___Common_rust_' 'Corn_(maize)___Northern_Leaf_Blight'
  'Corn_(maize)___healthy']

import tensorflow as tf

def prepare(file):
    IMG_SIZE = 256
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("cnn.model")

image = "corn\val\Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/0a01cc10-3892-4311-9c48-0ac6ab3c7c43___RS_GLSp 9352_new30degFlipLR.JPG"
prediction = model.predict([image])
prediction = list(prediction[0])
print(CATEGORIES[prediction.index(max(prediction))])
