# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:34:14 2017

@author: carl
"""
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

#from _future_ import print_function
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import keras
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
sess = tf.Session(config=config)
K.set_session(sess)
#import tensorflow as tf
#from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Convolution2D ,Flatten ,Dense, MaxPooling2D, BatchNormalization, LSTM, GRU, Bidirectional, TimeDistributed, concatenate, Lambda
from keras.utils import np_utils
from keras.optimizers import SGD ,Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers
from keras.models import load_model
from keras.layers.embeddings import Embedding


EMBEDDING_DIM = 128


train = pd.read_csv('train.csv', sep=',', header=0)
test = pd.read_csv('test.csv', sep=',', header=0)
movies = pd.read_csv('movies.csv', sep='::', header=0)
users = pd.read_csv('users.csv', sep='::', header=0)

User_ID = train.values.T[1]
Movie_ID = train.values.T[2]
Rating = train.values.T[3]

indices = np.arange(train.shape[0])
np.random.shuffle(indices)
User_ID = User_ID[indices]
Movie_ID = Movie_ID[indices]
Rating = Rating[indices]

User_ID = np.array(User_ID).astype(int)
Movie_ID = np.array(Movie_ID).astype(int)
Rating = np.array(Rating).astype('float32')

n_movies = 3883
n_users = 6040
num = len(Movie_ID)

User_ID =User_ID.reshape((num,1))
Movie_ID =Movie_ID.reshape((num,1))
Rating =Rating.reshape((num,1))

movie_input = keras.layers.Input(shape=[1])
movie_vec =keras.layers.Embedding(n_movies + 1, EMBEDDING_DIM)(movie_input)
movie_vec =keras.layers.Flatten()(movie_vec)

user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Embedding(n_users + 1, EMBEDDING_DIM)(user_input)
user_vec = keras.layers.Flatten()(user_vec)

Out = keras.layers.Dot(axes=1)([user_vec,movie_vec])
#Out = Dense(1)(Out)
#Out = Dropout(0.5)(Out)

movie_bias = keras.layers.Embedding(n_movies + 1, 1)(movie_input)
movie_bias = keras.layers.Flatten()(movie_bias)
#movie_bias = Dense(1)(movie_bias)
#movie_bias = Dropout(0.5)(movie_bias)

user_bias = keras.layers.Embedding(n_users + 1, 1)(user_input)
user_bias = keras.layers.Flatten()(user_bias)
#user_bias = Dense(1)(user_bias)
#user_bias = Dropout(0.5)(user_bias)

Out = keras.layers.Add()([Out,user_bias,movie_bias])


#Out = Dense(1)(Out)
#Out = Dropout(0.5)(Out)

#Out = Dense(1)(Out)
#Out = Dense(1)(Out)

model = keras.models.Model([user_input, movie_input], Out)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#		     metrics=['accuracy'])
print(model.summary())
#

filepath="hw5_1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
#
callbacks_list = [checkpoint]
model.fit([User_ID,Movie_ID],Rating-3,validation_split = 0.1, epochs=30, batch_size=2000,callbacks=callbacks_list)
tes_1 = test.values.T[1]
tes_2 = test.values.T[2]
model = load_model("hw5_1.hdf5")
adam = optimizers.Adam(lr=1e-4)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#		     metrics=['accuracy'])
predict = model.predict([tes_1,tes_2], batch_size=128)
predict = predict+3.0
with open("result2.csv", 'w') as f:
    f.write('TestDataID,Rating\n')
    for i, v in  enumerate(predict):
        f.write('%d,%f\n' %(i+1, v))