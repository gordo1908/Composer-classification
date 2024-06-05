# Composer Classification

This repository contains... 

## Training data and featurization

The training data consists of .mid files of musical compositions. A .mid file is a representation of the musical score. Several short snippets were extracted from the .mid files. The libfmp Python package (https://github.com/meinardmueller/libfmp) was used to read the .mid files and transform them to a piano roll-like representation of the snippets. Then note activity was sampled in time to create an input feature map (2D matrix). 10 second snippets were used with a sampling rate of 100 samples per second. Visual representations of an example piano roll and input feature map are given in data/piano_roll.png and data/input_feature_map.png respectively. \

The .mid files were obtained from kunstderfuge.com (https://www.kunstderfuge.com), a classical music database of .mid files. The model was trained to identify the compositions of two composers: Johann Sebastian Bach and Claude Debussy. The training set consisted of 397 .mid files and 10 samples per file and 13 samples per file were taken from the Bach and Debussy files respectively, to produce 2262 Bach and 2230 Debussy training samples. The test set consisted of 500 Bach and 500 Debussy samples. All the test samples were obtained from musical compositions not represented in the training dataset. 


## Model
A convolutional neural network was used. The final trained model had a training accuracy of 0.9983 and a test accuracy of 0.9400.
