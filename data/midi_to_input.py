# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 14:16:17 2021

@author: AG
"""

import os
import sys

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import colors
#import pretty_midi
import pandas as pd
import IPython.display as ipd
import numpy as np
import io
import os
import glob

#sys.path.append('..')
import libfmp.c1
#import music21 as m21


import pickle

np.set_printoptions(threshold=sys.maxsize)

def create_samplelist(filename, num_samples):
    fn = filename
    start_list = []
    score = libfmp.c1.midi_to_list(fn)
    #libfmp.c1.visualize_piano_roll(score, figsize=(8, 3), velocity_alpha=True)
    for note in score:
        start_list.append(note[0])
        
    start = min(start_list)
    last = max(start_list)
    sample_list = []
    for i in range(num_samples):
        sample_start = start + ((last-start)/(num_samples+1))*i
        sample_list.append(sample_start)
  
    return sample_list, score

def create_rep(filename,input_data,start,score):    
    #libfmp.c1.visualize_piano_roll(score, figsize=(8, 3), velocity_alpha=True);
    short_score = []
    end = 9.99 + start
    for i in score:
        if start <= i[0] < end:
            short_score.append(i)

    #libfmp.c1.visualize_piano_roll(short_score, figsize=(8, 3), velocity_alpha=True);

    time_slices=np.arange(start,end,0.01)

    rep = np.zeros((80,1000))
    for i in range(20,109):
        intervals = []
        for j in short_score:
            if j[2]==i:
                time_min = j[0]
                time_max = time_min + j[1]
                intervals.append([time_min,time_max])
 
        count = -1
        for k in time_slices:
            count = count + 1
            for l in intervals:
                if l[0] <= k <= l[1]:
                    #print("true",count)
                    rep[int(i-30),int(count)] = 1
                #print("true")
    #plt.imshow(rep,aspect='auto', origin='lower', interpolation='none')
    #plt.xlabel("Time (centiseconds)")
    #plt.ylabel("Pitch")
    #plt.show()
    input_data.append(rep)
    return input_data

##Specify folder where MIDI files are located##
file_list = glob.glob("music_database/Midi_files_1/bach/test/*.mid")
print(file_list)

##Specify number of samples to be made from each file##
num_samples = 10

input_data = []

##Loop through the MIDI files to create the input features##
for i in file_list:
    print(i)
    filename = i    
    sample_list, score = create_samplelist(filename, num_samples) 
    for start in sample_list:
        input_data = create_rep(filename, input_data,start,score)

train_X = np.stack(input_data)

print(train_X.shape)

##Save input to file##
pickle.dump(train_X, open("music_class/100samplespersecond/trial3/bach_test.pkl", "wb"))