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
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Concatenate, Add
from keras.optimizers import Adam
from keras.regularizers import l2 as l2_reg
from keras.initializers import Constant as const_init

def fast_random_bool(shape):
    n = np.prod(shape)
    nb = -(-n // 8)     # ceiling division
    b = np.fromstring(np.random.bytes(nb), np.uint8, nb)
    return np.unpackbits(b)[:n].reshape(shape).view(np.bool)

def read_images(filenames, shape=(80,320,3)):
    ret = np.zeros((len(filenames),) + shape)
    for i,f in enumerate(filenames):
        buf = mpimg.imread(f)
        ret[i] = buf[55:135,:,:]
        ret[i] = np.clip((ret[i]-np.mean(ret[i]))/np.std(ret[i]),-10,10)
    return ret

def image_generator(Xnames, Xsides, y, batch_size=128, use_shuffle=True, use_hflip=True):
    num_samples = len(Xnames)
    while 1:
        if use_shuffle:
            Xs, Ss, ys = shuffle(Xnames, Xsides, y)
        else:
            Xs, Ss, ys = Xnames, Xsides, y
        for offset in range(0, num_samples, batch_size):
            bs = min(batch_size, num_samples-offset)
            X_batch = read_images(Xs[offset:offset+batch_size])
            S_batch = Ss[offset:offset+batch_size]
            y_batch = ys[offset:offset+batch_size]
            if use_hflip:
                flip_mask = fast_random_bool(bs)
                X_batch[flip_mask,:,:,:] = X_batch[flip_mask,:,::-1,:]
                S_batch = S_batch * (1-2*flip_mask)
                y_batch = y_batch * (1-2*flip_mask)
            yield [X_batch, S_batch], y_batch

def get_session(gpu_fraction=0.8):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def conv_block(inputs, kernel, output_dims, reg=None):
    conv = Conv2D(output_dims, kernel, padding='same', activation='relu', \
            kernel_regularizer=None if reg is None else l2_reg(reg), \
            bias_regularizer=None if reg is None else l2_reg(reg))(inputs)
    conv = Conv2D(output_dims, kernel, padding='same', activation='relu', \
            kernel_regularizer=None if reg is None else l2_reg(reg), \
            bias_regularizer=None if reg is None else l2_reg(reg))(conv)
    return MaxPooling2D()(conv)

def atan_layer(x, **args):
    from keras import backend as K
    return K.tf.atan(x, **args)

def tan_layer(x, **args):
    from keras import backend as K
    return K.tf.tan(x, **args)

print('Initializing')
KTF.set_session(get_session())
log_pathnames = ['/dev/shm/data-track1/', '/dev/shm/data-track2/']
log_filename = 'driving_log.csv'
has_header = False
is_relative = False
dfs = []
for path_name in log_pathnames:
    if has_header:
        dfs.append(pd.read_csv(path_name+log_filename))
    else:
        dfs.append(pd.read_csv(path_name+log_filename, header=None, \
            names=['center','left','right','steering','throttle','brake','speed']))
if not is_relative:
    for path_name,df in zip(log_pathnames,dfs):
        for col in ['center','left','right']:
            df.loc[:,col] = df[col].apply(lambda s: path_name+s[s.index('IMG'):])

mult_factor = 10.
dropout_rate = 0.5
reg_rate = 1e-6
lr = 1e-3
lr_decay_rate = 0.8
side_init = 0.05

dfs_train = []
dfs_valid = []
for df in dfs:
    df_train, df_valid = train_test_split(df, test_size=0.1)
    dfs_train.append(df_train)
    dfs_valid.append(df_valid)

df_train = pd.concat(dfs_train, ignore_index=True, axis=0)
print(df_train.shape, ' '.join(str(df.shape) for df in dfs_valid))

Xs_valid, ys_valid = [], []
for df in dfs_valid:
    Xs_valid.append(list(df['center']))
    ys_valid.append(df['steering'].as_matrix())


X_train = list(df_train['center'])
y_train = df_train['steering'].as_matrix()
X_sides = np.zeros((len(X_train),))

left_names = list(df_train['left'])
X_train += left_names
X_sides = np.concatenate([X_sides, -np.ones((len(left_names),))])

right_names = list(df_train['right'])
X_train += right_names
X_sides = np.concatenate([X_sides, np.ones((len(right_names),))])

y_train = np.concatenate([y_train, y_train, y_train])


print('Building model')
input_layer = Input(shape=(80,320,3))
input_sides = Input(shape=(1,))
side_factor = Dense(1, use_bias=False, \
        kernel_initializer=const_init(side_init))(input_sides)

conv1 = conv_block(input_layer, 3, 16, reg_rate)
conv2 = conv_block(conv1, 5, 32, reg_rate)
conv3 = conv_block(conv2, 5, 64, reg_rate)

conv4_c = conv_block(conv3, 5, 128, reg_rate)
conv4_p = MaxPooling2D()(conv3)
conv4 = Concatenate()([conv4_c, conv4_p])

flatten = Flatten()(conv4)
hidden_1 = Dropout(dropout_rate)(flatten)
angle_layer = Dense(1, \
        kernel_regularizer=None if reg_rate is None else l2_reg(reg_rate), \
        bias_regularizer=None if reg_rate is None else l2_reg(reg_rate))(hidden_1)
angle_scaling = Dense(1, use_bias=False, trainable=False, \
        kernel_initializer=const_init(1/mult_factor*(np.pi*25/180)))(angle_layer)
tan_angle = Lambda(tan_layer)(angle_scaling)
tan_final = Add()([tan_angle,side_factor])
angle_final = Lambda(atan_layer)(tan_final)
output_layer = Dense(1, use_bias=False, trainable=False, \
        kernel_initializer=const_init(mult_factor/(np.pi*25/180)))(angle_final)

model = Model(inputs=[input_layer,input_sides], outputs=output_layer)
model.summary()

print(((y_train*mult_factor)**2).mean(), \
        ' '.join(str(((y_valid*mult_factor)**2).mean()) for y_valid in ys_valid))

train_batch_size = 128
train_generator = image_generator(X_train, X_sides, y_train*mult_factor, \
        batch_size=train_batch_size, \
        use_shuffle=True, use_hflip=True)
train_steps_per_epoch = np.ceil(len(X_train)/train_batch_size).astype(int)
valid_batch_size = 128
valid_generators = [image_generator(X_valid, np.zeros((len(X_valid),)), y_valid*mult_factor, \
        batch_size=valid_batch_size, \
        use_shuffle=False, use_hflip=False) for X_valid,y_valid in zip(Xs_valid,ys_valid)]
valid_steps_per_epoch = [np.ceil(len(X_valid)/valid_batch_size).astype(int) for X_valid in zip(Xs_valid)]

schedule = [5] * 10
tot = 0
for nb_epochs in schedule:
    opt = Adam(lr = lr)
    model.compile(opt, 'mse')
    model.fit_generator(train_generator, train_steps_per_epoch, epochs=nb_epochs)
    vals = [model.evaluate_generator(valid_generator, valid_step_per_epoch) \
            for valid_generator, valid_step_per_epoch in zip(valid_generators, valid_steps_per_epoch)]
    print(vals)
    lr *= lr_decay_rate
    tot += nb_epochs
    model.save('model-train-{0}.h5'.format(tot))
    print('Finished {0} epochs'.format(tot))

