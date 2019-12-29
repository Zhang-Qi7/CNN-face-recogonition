'''
    Load data from ORL dataset

    Please rename the pictures in the form:
    
    'si_j.bmp'
    
    where i is the label,j the number of faces
    
'''


from PIL import Image

from keras.preprocessing.image import load_img, save_img, img_to_array

import numpy as np

from six.moves import range

from keras import backend as K

'''
def load_data(img_nrows=92, img_ncols=92):

    if K.image_data_format() == 'channels_first':

        x_train = np.zeros(shape=(360, 1, img_nrows, img_ncols), dtype='float32')

        x_test = np.zeros(shape=(40, 1, img_nrows, img_ncols), dtype='float32')

    else:

        x_train = np.zeros(shape=(360, img_nrows, img_ncols, 1), dtype='float32')

        x_test = np.zeros(shape=(40, img_nrows, img_ncols, 1), dtype='float32')

    y_train = np.zeros(shape=(360,), dtype='int32')

    y_test = np.zeros(shape=(40,), dtype='int32')

    for i in range(40):

        for j in range(9):

            img_path = '.\ORL\s{:}_{:}.bmp'.format(i+1,j+1)

            img = Image.open(img_path)

            img = img.resize((img_nrows, img_ncols),Image.ANTIALIAS)

            img = img.convert('L')

            img = img_to_array(img)

            x_train[i*9+j, :, :, :] = img

            y_train[i*9+j] = i

        img_path = '.\ORL\s{:}_{:}.bmp'.format(i+1,10)

        img = Image.open(img_path)

        img = img.resize((img_nrows, img_ncols),Image.ANTIALIAS)

        img = img.convert('L')

        img = img_to_array(img)

        x_test[i, :, :, :] = img

        y_test[i] = i

    return (x_train, y_train), (x_test, y_test)
'''

def load_data(img_nrows=92, img_ncols=92):

    if K.image_data_format() == 'channels_first':

        x_train = np.zeros(shape=(360, 3, img_nrows, img_ncols), dtype='float32')

        x_test = np.zeros(shape=(40, 3, img_nrows, img_ncols), dtype='float32')

    else:

        x_train = np.zeros(shape=(360, img_nrows, img_ncols, 3), dtype='float32')

        x_test = np.zeros(shape=(40, img_nrows, img_ncols, 3), dtype='float32')

    y_train = np.zeros(shape=(360,), dtype='int32')

    y_test = np.zeros(shape=(40,), dtype='int32')

    for i in range(40):

        for j in range(9):

            img_path = '.\ORL\s{:}_{:}.bmp'.format(i+1,j+1)

            img = load_img(img_path, target_size=(img_nrows, img_ncols))

            img = img_to_array(img)

            x_train[i*9+j, :, :, :] = img

            y_train[i*9+j] = i

        img_path = '.\ORL\s{:}_{:}.bmp'.format(i+1,10)

        img = load_img(img_path, target_size=(img_nrows, img_ncols))

        img = img_to_array(img)

        x_test[i, :, :, :] = img

        y_test[i] = i

    return (x_train, y_train), (x_test, y_test)
