from keras.models import Sequential
import keras
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from math import floor
import pandas as pd
import numpy as np 

def normalize(X_train_test):
#    mu = (sum(X_train_test) / X_train_test.shape[0])
#    sigma = np.std(X_train_test, axis=0)
#    X_train_test_normed = (X_train_test - mu) / sigma
    X_train_test_normed = (X_train_test)/255
    return X_train_test_normed
def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])
def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))
    
    #X_all, Y_all = _shuffle(X_all, Y_all)
    
    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    
    return X_train, Y_train, X_valid, Y_valid
def load_data(train_data_path, test_data_path2):
    samples = pd.read_csv(train_data_path, sep=',')
    trainr = samples.values
    pics = np.zeros((len(trainr),48,48,1))
    feature = trainr[:,-1]

    for i, pic in enumerate(feature) :
        picp = np.array(pic.split()).astype('int')
        picp2 = normalize(picp)
        tmp = np.array(picp2.reshape(48,48,1))
        pics[i]=tmp
    label = trainr[:,0].astype('int')
    Y_train = np_utils.to_categorical(label,7)

    testr = pd.read_csv(test_data_path2, sep=',')
    X_test = np.zeros((len(testr),48,48,1))
    test = testr.values

    featuret = test[:,-1]
    for i, pict in enumerate(featuret) : 
        picpt = np.array(pict.split()).astype('int')
        picp2t = normalize(picpt)
        tmp2 = np.array(picp2t.reshape(48,48,1))
        X_test[i]=tmp2

    return (pics, Y_train, X_test)
#X_all, Y_all, X_test = load_data('train.csv','test.csv')
X_all, Y_all, X_test = load_data(sys.argv[1],sys.argv[2])
# Split a 10%-validation set from the training set
valid_set_percentage = 0.1
X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.250))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3), padding='same'))
#model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.250))

model.add(Flatten())
model.add(Dense(666))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(689))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(7))
model.add(BatchNormalization())
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)
epochs=1200
batch_size=64
model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size),
                    steps_per_epoch=int(np.ceil(X_train.shape[0] / float(batch_size))),
                    epochs=epochs,
                    validation_data=(X_valid, Y_valid),
                    workers=4)

loss_and_met = model.evaluate(X_train, Y_train, batch_size=128)
loss_and_metrics = model.evaluate(X_valid, Y_valid, batch_size=128)
classes = model.predict(X_test, batch_size=128)
from numpy import argmax
output_path = 'result.csv'
result = np.zeros(len(classes,))
for i, j in enumerate(classes):
    result[i] = argmax(j)
with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(result):
            f.write('%d,%d\n' %(i, v))
print('\n Train Acc:',loss_and_met[1])
print('\n Test Acc:',loss_and_metrics[1])

model.save('my_model_train.h5') 

## serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("model.h5")
print("Saved model to disk")