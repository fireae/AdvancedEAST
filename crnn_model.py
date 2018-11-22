from typing import List, Any, Union

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import plot_model
from cfg_crnn import *

from tensorflow.keras import backend as K
from tensorflow.python.keras.engine.base_layer import DeferredTensor

K.set_learning_phase(0)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_cnn(img_w, img_h, max_text_len, is_training):
    input_shape = (img_w, img_h, 1)

    # Make Networkw
    inputs = Input(name='input_image', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)
    # Convolution layer (VGG)
    inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(
        inputs)  # (None, 128, 64, 64)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

    inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(
        inner)  # (None, 64, 32, 128)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

    inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(
        inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(
        inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

    inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(
        inner)  # (None, 32, 8, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

    inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(
        inner)  # (None, 32, 4, 512)
    inner = BatchNormalization()(inner)
    x = Activation('relu')(inner)

    x = Reshape(target_shape=(32, 2048), name='reshape')(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(x)

    lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(x)
    lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm_1b')(x)
    lstm_1_merged = Add()([lstm_1, lstm_1b])
    lstm_1_merged = BatchNormalization()(lstm_1_merged)
    x = Dense(num_classes, kernel_initializer='he_normal', name='dense2')(lstm_1_merged)
    y_pred = Activation('softmax', name='softmax')(x)

    labels = Input(name='labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    if is_training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    else:
        return Model(inputs=[inputs], outputs=y_pred)


if __name__ == '__main__':
    model = build_cnn(128, 64, 10, True)
    print(model.summary())
    plot_model(model, 'crnn.png')
