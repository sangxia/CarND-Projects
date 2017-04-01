import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam

def read_images(filenames, shape=(80,320,3)):
    ret = np.zeros((len(filenames),) + shape)
    for i,f in enumerate(filenames):
        buf = mpimg.imread(f)
        ret[i] = buf[55:135,:,:]
    return ret

def image_generator(Xnames, y, batch_size=128, use_shuffle=True):
    num_samples = len(Xnames)
    while 1:
        if use_shuffle:
            Xs, ys = shuffle(Xnames, y)
        else:
            Xs, ys = Xnames, y
        for offset in range(0, num_samples, batch_size):
            yield read_images(Xs[offset:offset+batch_size]), ys[offset:offset+batch_size]

def get_session(gpu_fraction=0.8):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def conv_block(inputs, kernel, output_dims):
    conv = Conv2D(output_dims, kernel, padding='same', activation='relu')(inputs)
    conv = Conv2D(output_dims, kernel, padding='same', activation='relu')(conv)
    return MaxPool2D()(conv)

print('Initializing')
KTF.set_session(get_session())
log_pathname = '/dev/shm/newdata/'
log_filename = log_pathname + 'driving_log.csv'
has_header = False
is_relative = False
if has_header:
    df = pd.read_csv(log_filename)
else:
    df = pd.read_csv(log_filename, header=None, \
            names=['center','left','right','steering','throttle','brake','speed'])
if not is_relative:
    for col in ['center','left','right']:
        df.loc[:,col] = df[col].apply(lambda s: s[s.index('IMG'):])

center_names = list(log_pathname + df['center'])
X_train, X_valid, y_train, y_valid = train_test_split(center_names, df['steering'].as_matrix(), test_size=0.2)

adjust_angle = 5e-3

left_names = list(log_pathname + df['left'])
X_train += left_names
y_train = np.concatenate([y_train, (df['steering']+adjust_angle).as_matrix()])
right_names = list(log_pathname + df['right'])
X_train += right_names
y_train = np.concatenate([y_train, (df['steering']-adjust_angle).as_matrix()])

print(df['steering'].describe())

print('Building model')
input_layer = Input(shape=(80,320,3))
centered_input_layer = Lambda(lambda x: x/127.5-1., output_shape=(80,320,3))(input_layer)
conv1 = conv_block(centered_input_layer, 3, 16)
conv2 = conv_block(conv1, 5, 32)
conv3 = conv_block(conv2, 5, 64)
flatten = Flatten()(conv3)
hidden_1 = Dropout(0.8)(flatten)
output_layer = Dense(1)(hidden_1)
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

mult_factor = 10
print(((y_train*mult_factor)**2).mean(), ((y_valid*mult_factor)**2).mean())

train_batch_size = 128
train_generator = image_generator(X_train, y_train*mult_factor, batch_size=train_batch_size)
train_steps_per_epoch = np.ceil(len(X_train)/train_batch_size).astype(int)
valid_batch_size = 128
valid_generator = image_generator(X_valid, y_valid*mult_factor, batch_size=valid_batch_size, use_shuffle=False)
valid_steps_per_epoch = np.ceil(len(X_valid)/valid_batch_size).astype(int)

schedule = [(5,1e-3)]
for nb_epochs, lr in schedule:
    opt = Adam(lr = lr)
    model.compile(opt, 'mse')
    model.fit_generator(train_generator, train_steps_per_epoch, epochs=nb_epochs, \
            validation_data=valid_generator, validation_steps=valid_steps_per_epoch)

print('Saving model')
model.save('model.h5')

print('End')
