from keras.layers import *
from keras.models import Model
from keras.layers.normalization import *
from keras.regularizers import *
from keras import backend as K

import numpy as np

MAX_FILTERS_UNTIL_COMPRESSION = 784


def _create_compression_block(x, layer_id, compression=0.5, weight_decay=0.0001):
    while x.get_shape()[3] > MAX_FILTERS_UNTIL_COMPRESSION:
        filt = x.get_shape()[3]
        x = _compression_cell(x, layer_id, compression=compression, weight_decay=weight_decay)
        print('Compressing  %d -> %d' %(filt, x.get_shape()[3]))
    return x


def _create_cell_block(x, prev_x, n_channels, dropout, compression, layer_id, weight_decay=0.0001, N=1):
    for i in range(N):
        saved_x = x
        x = _normal_cell(x, prev_x, n_channels, layer_id)
        x = _create_compression_block(x, layer_id,
                                      compression=compression,
                                      weight_decay=weight_decay)
        if dropout > 0:
            x = Dropout(dropout)(x)
        prev_x = saved_x
    return x, prev_x


def _create_reduction_cell_block(x, prev_x, n_channels, dropout, layer_id):
    x = AveragePooling2D((2,2), padding='same')(x)
    prev_x = AveragePooling2D((2,2), padding='same')(prev_x)
    saved_x = x
    x = _reduction_cell(x, prev_x, n_channels, layer_id)
    if dropout > 0:
        x = Dropout(dropout)(x)
    prev_x = saved_x
    return x, prev_x

def _compression_cell(x, layer_id, compression=0.5, weight_decay=0.0001):
    layer_id['compression'] += 1
    uid = layer_id['compression']
    input_filters = int(x.get_shape()[3])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(int(input_filters*compression), (1,1), padding='same',
                           kernel_regularizer=l2(weight_decay),
                           kernel_initializer='he_normal',
                           name='compression-%s' % uid)(x)
    return x

def _normal_cell(x, prev_x, n_channels, layer_id):
    layer_id['normal'] += 1
    uid = layer_id['normal']

    # 'normal' cell
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    prev_x = BatchNormalization()(prev_x)
    prev_x = Activation('relu')(prev_x)

    s_1 = SeparableConv2D(n_channels, (3,3), padding='same', name='normal_s_1-%s' % uid)(x)
    # Supposed to be identity in s_2 but we need to dimensionality reduce to align for Add
    s_2 = x
    s_o = _add_with_matched_dims(s_1, s_2)

    t_1 = SeparableConv2D(n_channels, (3,3), padding='same', name='normal_t_1-%s' % uid)(prev_x)
    t_2 = SeparableConv2D(n_channels, (5,5), padding='same', name='normal_t_2-%s' % uid)(x)
    t_o = Add()([t_1, t_2])

    u_1 = AveragePooling2D((3,3), padding='same')(x)
    u_2 = prev_x
    u_1 = _upsample_layer(u_1, u_2)
    u_o = _add_with_matched_dims(u_1, u_2)

    v_1 = AveragePooling2D((3,3), padding='same')(prev_x)
    v_2 = AveragePooling2D((3,3), padding='same')(prev_x)
    v_o = Add()([v_1, v_2])
    v_o = _upsample_layer(v_o, u_o)

    c_1 = SeparableConv2D(n_channels, (5,5), padding='same', name='normal_c_1-%s' % uid)(prev_x)
    c_2 = SeparableConv2D(n_channels, (3,3), padding='same', name='normal_c_2-%s' % uid)(prev_x)
    c_o = Add()([c_1, c_2])

    x = Concatenate(axis=-1)([s_o, t_o, u_o, v_o, c_o])
    return x


def _reduction_cell(x, prev_x, n_channels, layer_id):
    layer_id['reduction'] += 1
    uid = layer_id['reduction']

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    prev_x = BatchNormalization()(prev_x)
    prev_x = Activation('relu')(prev_x)

    s_1 = SeparableConv2D(n_channels, (7,7), padding='same', name='reduction-s_1-%s' % uid)(prev_x)
    s_2 = SeparableConv2D(n_channels, (5,5), padding='same', name='reduction-s_2-%s' % uid)(x)
    s_3 = Add()([s_1, s_2])
    s_4 = MaxPooling2D((3,3), padding='same')(x)
    s_5 = SeparableConv2D(n_channels, (3,3), padding='same', name='reduction-s_5-%s' % uid)(s_3)
    s_6 = AveragePooling2D((3,3), padding='same')(s_3)
    s_4 = _upsample_layer(s_4, s_5)
    s_o = _add_with_matched_dims(s_4, s_5)

    t_1 = MaxPooling2D((3,3), padding='same')(x)
    t_2 = SeparableConv2D(n_channels, (7,7), padding='same', name='reduction-t_2-%s' % uid)(prev_x)

    t_1 = _upsample_layer(t_1, t_2)
    t_3 = _add_with_matched_dims(t_1, t_2)

    s_6 = _upsample_layer(s_6, t_3)
    t_o = Add()([s_6, t_3])

    u_1 = AveragePooling2D((3,3), padding='same')(x)
    u_2 = SeparableConv2D(n_channels, (5,5), padding='same', name='reduction-u_2-%s' % uid)(prev_x)

    u_1 = _upsample_layer(u_1, u_2)
    u_o = _add_with_matched_dims(u_1, u_2)

    x = Concatenate(axis=-1)([s_o, t_o, u_o])

    return x



def _upsample_layer(x, to_layer, factor=3):
    x = UpSampling2D((factor, factor))(x)
    crop_x = int(x.get_shape()[1] - to_layer.get_shape()[1])
    crop_y = int(x.get_shape()[2] - to_layer.get_shape()[2])
    if crop_x > 0 or crop_y > 0:
        x = Cropping2D(cropping=((crop_x, 0),(0, crop_y)))(x)
    return x


def _add_with_matched_dims(x, y):
    if int(x.get_shape()[3]) < int(y.get_shape()[3]):
        y = SeparableConv2D(int(x.get_shape()[3]), (1,1), padding='same')(y)
    elif int(x.get_shape()[3]) > int(y.get_shape()[3]):
        x = SeparableConv2D(int(y.get_shape()[3]), (1,1), padding='same')(x)
    return Add()([x, y])



def _create_dense_block(x, prev_x, n_layers, dropout, weight_decay,
                        compression, layer_id, max_filt, min_filt):
    x_list = [x]
    for i in range(n_layers):
        y, prev_x = _create_cell_block(x, prev_x,
            _convblock_filters(i, n_layers, max_filt=max_filt, min_filt=min_filt),
             dropout, compression, layer_id, N=1)
        x_list.append(y)
        x = Concatenate(axis=-1)(x_list)
        unscaled_x = y
        if i == n_layers - 1:
            y, prev_x = _create_reduction_cell_block(x, prev_x,
                _convblock_filters(i, n_layers, max_filt=max_filt,
                    min_filt=min_filt),
                 dropout, layer_id)
        output_size = _convblock_filters(i, n_layers, max_filt=max_filt, min_filt=min_filt)
        # nedanstående ger lite sämre acc men känns mer logiskt
        #output_size = int(y.get_shape()[3])
    print('Filters: %s' % int(y.get_shape()[3]))
    return y, prev_x, unscaled_x, output_size

def _convblock_filters(i, n_layers, max_filt = 96, min_filt = 32):
    return max(min_filt+i, min(max_filt+n_layers, int((max_filt+n_layers)*float((i+1)**2.5/n_layers**2.5))))


def _create_transition_layer(x, n_channels, dropout, weight_decay, compression, layer_id, kernel=5):
    layer_id['transition'] += 1
    uid = layer_id['transition']
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    y = SeparableConv2D(int(n_channels/2), (kernel,kernel), padding='same', use_bias=False,
                      kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal', name='transition-large-%s' % uid)(x)
    x = SeparableConv2D(int(n_channels/2), (kernel-2,kernel-2), padding='same', use_bias=False,
                      kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal', name='transition-small-%s' % uid)(x)
    x = Concatenate(axis=-1)([y, x])
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = AveragePooling2D((2,2))(x)
    return x


def _create_initial_layer(x, n_channels):
    z = Convolution2D(n_channels, (5,5), kernel_initializer='he_normal', padding='same', name='initial-large')(x)
    y = Convolution2D(n_channels, (3,3), kernel_initializer='he_normal', padding='same', name='initial-medium')(x)
    x = Convolution2D(n_channels, (2,2), kernel_initializer='he_normal', padding='same', name='initial-small')(x)
    x = Concatenate(axis=-1)([z, y, x])
    return x


def _create_classification_layer(x, nbr_classes):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nbr_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return x


def create_densenet(input_shape, dense_layers, nbr_classes, weight_decay,
                    filters_per_channel=24, compression=0.5, dropout=0.2,
                    max_filt=96, min_filt=32):
    layer_id = {'transition': 0, 'normal': 0, 'reduction': 0, 'compression': 0}
    n_channels = 3
    x = Input(input_shape)
    prev_x = x
    y = _create_initial_layer(x, filters_per_channel)
    n_channels = filters_per_channel * 3
    concatenated_history = y
    dense_blocks = len(dense_layers)

    for i in range(dense_blocks):
        y, prev_x, unscaled_x, last_filt = _create_dense_block(y, prev_x,
                        dense_layers[i], dropout, weight_decay, compression, layer_id,
                        max_filt=max_filt, min_filt=min_filt)
        n_channels = n_channels + last_filt
        concatenated_history = Concatenate(axis=-1)([concatenated_history, unscaled_x])
        concatenated_history = _create_transition_layer(concatenated_history, n_channels, dropout, weight_decay,
                                     compression, layer_id, kernel = 6-i)
        if i == dense_blocks - 1:
            y = Concatenate(axis=-1)([concatenated_history, y])
            y = _create_classification_layer(y, nbr_classes)
        print(print('t: %d' % n_channels))
    model = Model(inputs=x, outputs=y, name='Semi-DenseNAS')
    return model
