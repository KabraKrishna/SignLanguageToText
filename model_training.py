# coding: utf-8


import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

from datetime import datetime
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
from sklearn.model_selection import train_test_split



os.listdir()


# Exploring Classes


dataset_path='dataset'
categories=os.listdir(dataset_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels)) 

print(label_dict)
print(categories)
print(labels)


data_path='dataset/train'
classes_path=os.listdir(data_path)
classesf=os.listdir(data_path)
print(classesf)
labels_classes=[i for i in range(len(classesf))]
print(labels_classes)


data_path='dataset'


label_classes_dict=dict(zip(classesf,labels_classes))
print(label_classes_dict)


data=[]
target=[]
c=0
minValue = 70
for category in categories:
    
    cat_path=os.path.join(data_path,category)
    print(cat_path)
    cat_names=os.listdir(cat_path)
    print(cat_names)
    for classes in cat_names:
        folder_path=os.path.join(data_path,category,classes)
        img_names=os.listdir(folder_path)
        for img_name in img_names:
            img_path=os.path.join(folder_path,img_name)
            img=cv2.imread(img_path)
            #applying image preprocessing
            try:
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # converting to gray Scale
                blur = cv2.GaussianBlur(gray,(5,5),2)      # applying Blur -> kernal size 5x5
                #applying adaptive threshold
                th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) 
                # applying threashold again for more prominent edges
                ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                
                resized=cv2.resize(res,(128,128)) # resizing  to 128 x 128
                data.append(resized)
                target.append(label_classes_dict[classes])
            except Exception as e:
                print('Exception:',e)


# Reference: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

# Saving array data


datanp=np.array(data)

datanp.shape



targetnp=np.array(target)

targetnp.shape



import numpy as np

data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],128,128,1))
target=np.array(target)

from keras.utils import np_utils

new_target=np_utils.to_categorical(target)
new_target.shape


np.save('data_img',data)
np.save('target',new_target)
print("\nData Saved Successfully.........")

print("\nLoading data.........")


data=np.load('data_img.npy')
target=np.load('target.npy')


# Train Test Split


train_data,test_data,train_target,test_target=train_test_split(data,new_target,test_size=0.2)


# Architecture
# 3 Convlution layers (32, 64)
# 3 Dense layers
# 1 Output layer


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.40))
model.add(Dense(units=96, activation='relu'))
model.add(Dropout(0.40))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=27, activation='softmax'))

print(model.summary())


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



from keras.callbacks import ModelCheckpoint



checkpoint = ModelCheckpoint('model_check_points/model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history= model.fit(train_data,train_target,shuffle=True,epochs=20,callbacks=[checkpoint],validation_split=0.3)


print(model.evaluate(test_data,test_target))


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(['train_loss','val_loss'], loc=0)
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend(['train_accuracy','val_accuracy'], loc=0)
plt.show()


model.save('model_new.h5')

