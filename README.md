# Composer Classification

This repository contains... 

## Training data and featurization

The training data is .mid files of pieces by Bach and Debussy. A .mid file contains a representation of the musical score, the essential production of a composer. Several short snippets were extracted from the .mid files. The libfmp Python package (https://github.com/meinardmueller/libfmp) was used to read the .mid files and transform them to a piano roll-like representation of the snippets. Then note activity was sampled in time to create an input feature map (2D matrix). 10 second snippets were used with a sampling rate of 100 samples per second. Visual representations of an example piano roll and input feature map are given in data/piano_roll.png and data/input_feature_map.png respectively. 

## Model
