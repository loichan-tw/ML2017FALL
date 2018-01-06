# -*- coding: utf-8 -*-
"""
@author: carl
"""
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from keras.models import Model
from keras.models import load_model
import sys
#load images
img = np.load(sys.argv[1])
img = img.astype('float32') / 255.
data_path = sys.argv[2]
# parameters
encoding_dim = 128 
epoch_num = 2000
batch_size_ = 400
filepath = 'hw6_.hdf5'
output_ = sys.argv[3]

autoencoder = load_model(filepath)
autoencoder.compile(loss='mse', optimizer='adam')
layer_name = 'layero'
intermediate_layer_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer_name).output)
code = intermediate_layer_model.predict(img)

test_data = pd.read_csv(data_path, sep=',',encoding='utf-8')
test = test_data.values
result = np.zeros((len(test),))
cluster = KMeans(n_clusters=2).fit(code)
class_ = cluster.labels_
for i, id_ in enumerate(test):
    temp1 = class_[id_[1]]
    temp2 = class_[id_[2]]
    result[i] = (temp1==temp2).astype(int)
with open(output_, 'w') as f:
    f.write('ID,Ans\n')
    for i, v in  enumerate(result):
        f.write('%d,%d\n' %(i, v))
#from keras.layers import Input, Dense
#from keras.callbacks import ModelCheckpoint, EarlyStopping
## model
#input_img = Input(shape=(784,))
#encoded = Dense(128, activation='relu')(input_img)
#encoded = Dense(64, activation='relu')(encoded)
#encoded = Dense(encoding_dim, activation='relu',name='layero')(encoded)
#
#decoded = Dense(64, activation='relu')(encoded)
#decoded = Dense(128, activation='relu')(decoded)
#decoded = Dense(784, activation='sigmoid')(decoded)
#autoencoder = Model(input_img, decoded)
#encoder = Model(input_img, encoded)
#encoded_input = Input(shape=(encoding_dim,))
#decoder_layer = autoencoder.layers[-1]
#decoder = Model(encoded_input, decoder_layer(encoded_input))
#autoencoder.compile(optimizer='adam', loss='mse')
#
## fit and save model
#early_stopping =EarlyStopping(monitor='loss', patience=5, mode='min')
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
#callbacks_list = [checkpoint, early_stopping]
#autoencoder.fit(img , img ,epochs=epoch_num, batch_size=batch_size_, shuffle=True,callbacks=callbacks_list)
