# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:24:07 2017

@author: carl
"""
import os, sys
import pandas as pd
import numpy as np
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#from keras import models
def normalize(X_train_test):
    X_train_test_normed = (X_train_test)/255
    return X_train_test_normed
def load_data(test_data_path):
    testr = pd.read_csv(test_data_path, sep=',')
    X_test = np.zeros((len(testr),48,48,1))
    test = testr.values

    featuret = test[:,-1]
    for i, pict in enumerate(featuret) : 
        picpt = np.array(pict.split()).astype('int')
        picp2t = normalize(picpt)
        tmp2 = np.array(picp2t.reshape(48,48,1))
        X_test[i]=tmp2
    return (X_test)
#X_test = load_data('test.csv')
X_test = load_data(sys.argv[1])
from keras.models import load_model
#loaded_model = load_model('my_model.h5')
loaded_model = load_model(sys.argv[3])
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
classes = loaded_model.predict(X_test, batch_size=128)
from numpy import argmax
#output_path = 'result_.csv'
output_path = sys.argv[2]
result = np.zeros(len(classes,))
for i, j in enumerate(classes):
    result[i] = argmax(j)
with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(result):
            f.write('%d,%d\n' %(i, v))
#from keras.models import model_from_json
##from keras.models import load_weights
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights('model.h5')
# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
