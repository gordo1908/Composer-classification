# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 15:17:51 2021

@author: AG
"""

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

reconstructed_model = tensorflow.keras.models.load_model("music_class/100samplespersecond/trial3/model_dropout")
infile = open('music_class/100samplespersecond/trial3/test.pkl','rb')
test_X = pickle.load(infile)

Y_list = []
test_Y1 = np.zeros((50))
Y_list.append(test_Y1)
test_Y2 = np.ones((50))
Y_list.append(test_Y2)
test_Y = np.concatenate(Y_list)

test_X = test_X.reshape(-1, 80,1000, 1)
print(test_X.shape)

test_X = test_X.astype('float32')
test_Y_one_hot = to_categorical(test_Y)

test_eval = reconstructed_model.evaluate(test_X, test_Y_one_hot, verbose=0)
ynew = reconstructed_model.predict_classes(test_X)

print(test_Y)
print(ynew)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])



