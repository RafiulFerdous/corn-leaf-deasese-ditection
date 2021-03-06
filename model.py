
"""
Created on Wed Jul 15 20:15:39 2020

@author: Admin
"""


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

#tf.debugging.set_log_device_placement(True)


#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print("NUm gpu available: ", len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


EPOCHS = 25
INIT_LR = 1e-3
BS = 64
default_image_size = tuple((256, 256))
image_size = 0
directory_root = 'normal/'
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
    
    
image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for disease_folder in plant_disease_folder_list :
            # remove .DS_Store from list
            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")
    
image_size = len(image_list)
    
    
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)

print(label_binarizer.classes_)


np_image_list = np.array(image_list, dtype=np.float16) / 225.0

print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 


aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")



model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))

model.summary()

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print("[INFO] training network...")

history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
   epochs=EPOCHS, verbose=1
  )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()


print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

# y_pred = model.predict(x_test)
# print(y_pred)
# y_pred = np.argmax(y_pred, axis=1)
# print(y_pred)


# y_pred = model.predict_classes(x_test)
# print(y_pred)

# p=model.predict_proba(x_test)

# target_name = ['class 0(Cercospora_leaf_spot Gray_leaf_spot', 'class 1(Common_rust)', 'class 2(healthy)', 'class 3(northern l)']
# print(classification_report(np.argmax(y_test, axis=1), y_pred, target_name=target_name))

# fname = "weight-modelcnn.hdf5"
# model.save_weights(fname, overwrite=True)

# fname = "weight-modelcnn.hdf5"
# model.load_weights(fname)
print("[INFO] Saving model...")
pickle.dump(model,open('cnn_model.pkl', 'wb'))


model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")

#Loading the saved model

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# lb = pickle.load(open("label_transform.pkl","rb"))

# INIT_LR = 1e-3
# opt=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# loaded_model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# img = cv2.imread('im1.jpg')
# img = cv2.resize(img,(256,256))
# img = ImageDataGenerator(
#     rotation_range=25, width_shift_range=0.1,
#     height_shift_range=0.1, shear_range=0.2, 
#     zoom_range=0.2,horizontal_flip=True, 
#     fill_mode="nearest")

# img = np.reshape(img,[1,256,256,3])

# classes = loaded_model.predict_classes(img)



# sA = sparse.csr_matrix(classes)
# lb = pickle.load(open("label_transform.pkl","rb"))
# disease = f"{lb.inverse_transform(sA)[0]}"
# print("The Disease detected is :",disease)

# img2 = cv2.imread('im1.jpg')
# plt.imshow(img2,)




# #FLOW1_model = load_model('model.h5')
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# lb = pickle.load(open("label_transform.pkl","rb"))


# #Confusion Matrix and Classification Report
# Y_pred = loaded_model.predict_generator(test_generator, test_generator.samples // test_generator.batch_size+1)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(test_generator.classes, y_pred))
# print('Classification Report')
# target_names = ['cl', 'cr', 'h','nl']
# print(classification_report(test_generator.classes, y_pred, target_names=target_names))


# #Evaluating using Keras model_evaluate:
    
    
# INIT_LR = 1e-3
# opt=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# loaded_model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
# # x, y = zip(*(test_generator[i] for i in range(len(test_generator))))
# # x_test, y_test = np.vstack(x), np.vstack(y)
# # loss, acc = loaded_model.evaluate(x_test, y_test, batch_size=32)

# # print("Accuracy: " ,acc)
# # print("Loss: ", loss)


# print(label_binarizer.classes_)
    
# loaded_model = pickle.load(open('cnn_model.pkl', 'rb'))

# model_disease=loaded_model

# image_dir="corn\val\Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/0a01cc10-3892-4311-9c48-0ac6ab3c7c43___RS_GLSp 9352_new30degFlipLR.JPG"

# im=convert_image_to_array(image_dir)
# np_image_li = np.array(im, dtype=np.float16) / 225.0
# npp_image = np.expand_dims(np_image_li, axis=1)

# result=model_disease.predict(npp_image)
# print(result)

# itemindex = np.where(result==np.max(result))
# print("probability:"+str(np.max(result))+"\n"+label_binarizer.classes_[itemindex[1][0]])





