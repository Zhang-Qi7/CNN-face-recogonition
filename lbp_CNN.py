'''
Train a simple CNN for face recogonition based on local binary pattern (lbp) transform

'''


from __future__ import print_function

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

import os

from keras import backend as K

import numpy as np

import ORL,YALE


batch_size = 20

num_classes = 40

epochs = 100

save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = 'keras_lbp_trained_model.h5'


(x_train, y_train), (x_test, y_test) = ORL.load_data(92, 92)

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)


def lbp_image(img):

    '''
    local binary pattern images of faces

    img: np.ndarray of the form (samples, rows, cols, channels) or (samples, rows, cols, channels)

    '''
    
    spoint = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[-0,1],[1,-1],[1,0],[1,1]])

    neighbor = 8

    minx = np.min(spoint[:,0])

    maxx = np.max(spoint[:,0])

    miny = np.min(spoint[:,1])

    maxy = np.max(spoint[:,1])


    bsizex = (np.ceil(np.max(maxx,0))-np.floor(np.min(minx,0))+1).astype('int32')

    bsizey = (np.ceil(np.max(maxy,0))-np.floor(np.min(miny,0))+1).astype('int32')

    originx = (0-np.floor(np.min(minx,0))).astype('int32')

    originy = (0-np.floor(np.min(miny,0))).astype('int32')


    if K.image_data_format() == 'channels_first':

        batch, channel, xsize, ysize = img.shape
        
        assert xsize > bsizex and ysize > bsizey

        dx = xsize - bsizex

        dy = ysize - bsizey

        result = np.zeros((batch, channel, dx+1, dy+1), dtype='float32')

        C = img[:, :, originx:originx+dx+1, originy:originy+dy+1]

        for i in range(neighbor):

            x = spoint[i,0]+originx

            y = spoint[i,1]+originy

            N = img[:, :, x:x+dx+1, y:y+dy+1]

            D = N > C

            v = 2 ** i

            result = np.add(result, v*D)
        
    else:

        batch, xsize, ysize, channel = img.shape
        
        assert xsize > bsizex and ysize > bsizey

        dx = xsize - bsizex

        dy = ysize - bsizey

        result = np.zeros((batch, dx+1, dy+1, channel), dtype='float32')

        C = img[:, originx:originx+dx+1, originy:originy+dy+1, :]

        for i in range(neighbor):

            x = spoint[i,0]+originx

            y = spoint[i,1]+originy

            N = img[:, x:x+dx+1, y:y+dy+1, :]

            D = N > C

            v = 2 ** i

            result = np.add(result, v*D)

    return result


x_train = lbp_image(x_train) / 255

x_test = lbp_image(x_test) / 255



model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:]))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(32, (3, 3), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))


model.add(Conv2D(32, (3, 3), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))


model.add(Flatten())

model.add(Dense(num_classes))

model.add(Activation('softmax'))


opt = keras.optimizers.Adadelta()

#opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])


model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))


if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)


scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])   

    
