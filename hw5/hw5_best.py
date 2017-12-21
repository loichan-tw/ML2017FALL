# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:05:49 2017

@author: carl
"""
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras.models import load_model
from keras import optimizers
import pandas as pd
import sys
test = pd.read_csv(sys.argv[1], sep=',', header=0)
tes_1 = test.values.T[1]
tes_2 = test.values.T[2]
model = load_model("hw5_1.hdf5")
adam = optimizers.Adam(lr=1e-4)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

predict = model.predict([tes_1,tes_2], batch_size=128)
predict = predict+3.0
with open(sys.argv[2], 'w') as f:
    f.write('TestDataID,Rating\n')
    for i, v in  enumerate(predict):
        f.write('%d,%f\n' %(i+1, v))