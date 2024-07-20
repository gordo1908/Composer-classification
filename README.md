# Composer Classification

In this project, we aim to use machine intelligence to identify the composers of short musical excerpts. The types of models explored in this work could serve as musicological tools in order to identify unique stylistic characteristics of composers and authenticate works of disputed authorship.


## Training data and featurization

The training data consists of MIDI (.mid) files downloaded from kunstderfuge.com (https://www.kunstderfuge.com). A MIDI file consists of a set of musical instructions that can be read by a computer, similar to how a musical score is read by a musician. 

## Objective and approach
The objective of this work is to develop a model that can classify musical examples taken from a database of compositions by two composers, Johann Sebastian Bach and Claude Debussy. We created 2D feature representations from MIDI files and used them to train convolutional neural networks. The details of featurization, model training and evaluation are described in composer-class-notebook.ipynb.

