# _*_ coding:utf-8 _*_
# time: 11/7/18 10:40 AM

import tensorflow as tf
import tensorflow.keras as keras

print(keras.__version__)

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras.utils import plot_model

import cfg


class East:
    def __init__(self):
        self.input_img = Input(name='input_img',
                               shape=(None, None, cfg.num_channels), dtype='float32')
        vgg16 = VGG16(input_tensor=self.input_img, weights='imagenet', include_top=False)
        if cfg.locked_layers:
            locked_layers = [vgg16.get_layer('block1_conv1'), vgg16.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        self.f = [vgg16.get_layer('block%d_pool' % i).output for i in cfg.feature_layers_range]
        self.f.insert(0, None)
        self.diff = cfg.feature_layers_range[0] - cfg.feature_layers_num

    def g(self, i):
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + str(cfg.feature_layers_range)
        if i == cfg.feature_layers_num:
            bn = BatchNormalization()(self.h(i))
            return Conv2D(32, 3, activation='relu', padding='same')(bn)
        else:
            return UpSampling2D((2, 2))(self.h(i))

    def h(self, i):
        if i == 1:
            return self.f[i]
        else:
            concat = Concatenate(axis=-1)([self.g(i - 1), self.f[i]])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(128 // 2 ** (i - 2), 1, activation='relu', padding='same')(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_3 = Conv2D(128 // 2 ** (i - 2), 3, activation='relu', padding='same')(bn2)
            return conv_3

    def build_network(self):
        before_input = self.g(cfg.feature_layers_num)
        inside_score = Conv2D(1, 1, padding='same', name='inside_core')(before_input)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code')(before_input)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord')(before_input)
        east_detect = Concatenate(axis=-1, name='east_detect')([inside_score,
                                                                side_v_code, side_v_coord])
        return Model(inputs=self.input_img, outputs=east_detect)


if __name__ == '__main__':
    east = East()
    east_network = east.build_network()
    east_network.summary()
    plot_model(east_network, to_file='model.png')
