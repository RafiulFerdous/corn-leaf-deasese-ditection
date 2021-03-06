# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 12:24:48 2020

@author: Rasel
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
from sklearn.metrics import classification_report,confusion_matrix






#image_size = 224
test_dir = 'corn/Val'
test_batchsize = 32
EPOCHS = 25


test_datagen = ImageDataGenerator(rescale=1./255, 
                                  rotation_range=25, width_shift_range=0.1,
                                  height_shift_range=0.1, shear_range=0.2, 
                                  zoom_range=0.2,horizontal_flip=True, 
                                  fill_mode="nearest")
test_generator = test_datagen.flow_from_directory( 
        test_dir,
        target_size=(256, 256),
        batch_size=test_batchsize,
        shuffle = False,
        class_mode='categorical'
   )

#FLOW1_model = load_model('model.h5')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
lb = pickle.load(open("label_transform.pkl","rb"))


#Confusion Matrix and Classification Report
Y_pred = loaded_model.predict_generator(test_generator, test_generator.samples // test_generator.batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['cl', 'cr', 'h','nl']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))


#Evaluating using Keras model_evaluate:
    
    
INIT_LR = 1e-3
opt=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
loaded_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
x, y = zip(*(test_generator[i] for i in range(len(test_generator))))
x_test, y_test = np.vstack(x), np.vstack(y)
loss, acc = loaded_model.evaluate(x_test, y_test, batch_size=32)

print("Accuracy: " ,acc)
print("Loss: ", loss)