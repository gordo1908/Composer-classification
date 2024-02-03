# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 13:11:55 2021

@author: AG
"""

#Import libraries
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import tensorflow.keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
import pickle

#Load the training data; concatenate the bach and debussy data
filename = ['music_class/100samplespersecond/trial2/debussy.pkl', 'music_class/100samplespersecond/trial2/bach.pkl']
complete_list = []
for i in filename:
    infile = open(i,'rb')
    reps = pickle.load(infile)
    complete_list.append(reps)
    infile.close()
    #print(reps.shape)
train_X = np.concatenate(complete_list)

#Create the training data labels
Y_list = []
train_Y1 = np.zeros((2230))
Y_list.append(train_Y1)
train_Y2 = np.ones((2230))
Y_list.append(train_Y2)
train_Y = np.concatenate(Y_list)

print('Training data shape : ', train_X.shape, train_Y.shape)
#print('Testing data shape : ', test_X.shape, test_Y.shape)

train_X = train_X.reshape(-1, 80,1000, 1)
print(train_X.shape)

train_X = train_X.astype('float32')
train_Y_one_hot = to_categorical(train_Y)

#Perform train/validation split
from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

#Define CNN model
batch_size = 100
epochs = 12
num_classes = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(80,1000,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))  
model.add(Dropout(0.3))                
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=tensorflow.keras.losses.binary_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(),metrics=['accuracy'])
#model.summary()

#Train CNN model
train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

#Save CNN model
model.save('music_class/100samplespersecond/trial2/model_dropout')