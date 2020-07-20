# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 11:33:42 2020

@author: bodda
"""

dataset=r'E:\covid xray DL\Data'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import skimage.transform
data=[]
labels=[]


for dir_path,dir_name,files in os.walk(dataset):
    for f in files:
        image=cv2.imread(os.path.join(dir_path,f))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(224,224))
        data.append(image)
        label=dir_path.split(os.path.sep)
        labels.append(label[-1])
        
data=np.array(data)/255.0
labels=np.array(labels)   
'''
#visualizing the images
Nimages=os.listdir(dataset+"//Normal")
Cimages=os.listdir(dataset+"//Covid")
img=cv2.imread(dataset+"//Normal//"+Nimages[1])
img=skimage.transform.resize(img,(150,150,3))
img2=cv2.imread(dataset+"//Covid//"+Cimages[1])
img2=skimage.transform.resize(img2,(150,150,3),mode='reflect')
pair=np.concatenate((img,img2),axis=1)
plt.show(img)
'''
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,stratify=labels)
from keras.preprocessing.image import ImageDataGenerator
trainAug=ImageDataGenerator(rotation_range=12,fill_mode='nearest')

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D,Flatten,Dropout,Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
base_model=VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))
base_model.summary()

hmodel=base_model.output
hmodel=AveragePooling2D(pool_size=(4,4))(hmodel)
hmodel=Flatten(name='flatten')(hmodel)
hmodel=Dropout(0.5)(hmodel)
hmodel=Dense(2,activation='softmax')(hmodel)
model=Model(base_model.input,hmodel)

for layers in base_model.layers:
    layers.trainable=False
model.summary()
from tensorflow.keras.optimizers import Adam
init_lr=1e-3
epochs=10
opt=Adam(lr=init_lr,decay=init_lr/epochs)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
model.fit_generator(trainAug.flow(x_train,y_train,batch_size=32),steps_per_epoch=len(x_train)//32,validation_data=(x_test,y_test),validation_steps=len(x_test)//32,epochs=30)
y_pred=model.predict(x_test,batch_size=32)
y_pred=np.argmax(y_pred,axis=1)
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test.argmax(axis=1),y_pred,target_names=lb.classes_))
print(accuracy_score(y_test.argmax(axis=1),y_pred))
l=6
w=5
fig,axes=plt.subplots(l,w,figsize=(12,12))
axes=axes.ravel()
y_pred=model.predict(x_test,batch_size=32)
for i in np.arange(0,l*w):
    axes[i].imshow(x_test[i])
    if y_test.argmax(axis=1)[i]:
        true_h="Corona"
    else:
        true_h="Normal"
    if y_pred.argmax(axis=1)[i]:
        pred_h="corona"
    else:
        pred_h="Normal"
    axes[i].set_title("prediction ={}\n True ={}".format(str(pred_h),str(true_h)))
    axes[i].axis('off')
plt.subplots_adjust(wspace=1,hspace=1)