# -*- coding: utf-8 -*-
"""DenseNet models for Keras.

# Reference paper

- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation

- [Torch DenseNets]
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras import backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.utils import plot_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import _obtain_input_shape

PATH_PREFIX = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/'
DENSENET121_WEIGHT_PATH = PATH_PREFIX + \
    'densenet121_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET121_WEIGHT_PATH_NO_TOP = PATH_PREFIX + \
    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
DENSENET169_WEIGHT_PATH = PATH_PREFIX + \
    'densenet169_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET169_WEIGHT_PATH_NO_TOP = PATH_PREFIX + \
    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
DENSENET201_WEIGHT_PATH = PATH_PREFIX + \
    'densenet201_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET201_WEIGHT_PATH_NO_TOP = PATH_PREFIX + \
    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'


def dense_block(input_tensor, blocks, name):
    """A dense block.

    # Arguments
        input_tensor: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        input_tensor = conv_block(input_tensor, 32, name=name + '_block' + str(i + 1))
    return input_tensor


def transition_block(input_tensor, reduction, name):
    """A transition block.

    # Arguments
        input_tensor: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    input_tensor = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(input_tensor)
    input_tensor = Activation('relu', name=name + '_relu')(input_tensor)
    input_tensor = Conv2D(int(K.int_shape(input_tensor)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv')(input_tensor)
    input_tensor = AveragePooling2D(2, strides=2, name=name + '_pool')(input_tensor)
    return input_tensor


def conv_block(input_tensor, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        input_tensor: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    layer = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(input_tensor)
    layer = Activation('relu', name=name + '_0_relu')(layer)
    layer = Conv2D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv')(layer)
    layer = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(layer)
    layer = Activation('relu', name=name + '_1_relu')(layer)
    layer = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(layer)
    output_layer = Concatenate(axis=bn_axis, name=name + '_concat')(
                                [input_tensor, layer])
    return output_layer


# Tied for Public Enemy #5 for too-many-branches
# pylint: disable=too-many-branches
# Public Enemy #8 for too-many-statements
# pylint: disable=too-many-statements
def dense_net(blocks,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the DenseNet architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with
    TensorFlow, Theano, and CNTK. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=221,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    layer = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    layer = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(layer)
    layer = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(layer)
    layer = Activation('relu', name='conv1/relu')(layer)
    layer = ZeroPadding2D(padding=((1, 1), (1, 1)))(layer)
    layer = MaxPooling2D(3, strides=2, name='pool1')(layer)

    layer = dense_block(layer, blocks[0], name='conv2')
    layer = transition_block(layer, 0.5, name='pool2')
    layer = dense_block(layer, blocks[1], name='conv3')
    layer = transition_block(layer, 0.5, name='pool3')
    layer = dense_block(layer, blocks[2], name='conv4')
    layer = transition_block(layer, 0.5, name='pool4')
    layer = dense_block(layer, blocks[3], name='conv5')

    layer = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(layer)

    if include_top:
        layer = GlobalAveragePooling2D(name='avg_pool')(layer)
        layer = Dense(classes, activation='softmax', name='fc1000')(layer)
    else:
        if pooling == 'avg':
            layer = GlobalAveragePooling2D(name='avg_pool')(layer)
        elif pooling == 'max':
            layer = GlobalMaxPooling2D(name='max_pool')(layer)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, layer, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, layer, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, layer, name='densenet201')
    else:
        model = Model(inputs, layer, name='densenet')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if blocks == [6, 12, 24, 16]:
                weights_path = get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET121_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='0962ca643bae20f9b6771cb844dca3b0')
            elif blocks == [6, 12, 32, 32]:
                weights_path = get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET169_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='bcf9965cf5064a5f9eb6d7dc69386f43')
            elif blocks == [6, 12, 48, 32]:
                weights_path = get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET201_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='7bb75edd58cb43163be7e0005fbe95ef')
        else:
            if blocks == [6, 12, 24, 16]:
                weights_path = get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET121_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='4912a53fbd2a69346e7f2c0b5ec8c6d3')
            elif blocks == [6, 12, 32, 32]:
                weights_path = get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET169_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='50662582284e4cf834ce40ab4dfa58c6')
            elif blocks == [6, 12, 48, 32]:
                weights_path = get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET201_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='1c2de60ee40562448dbac34a0737e798')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def dense_net_121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return dense_net([6, 12, 24, 16],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


def dense_net_169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return dense_net([6, 12, 32, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


def dense_net_201(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return dense_net([6, 12, 48, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


def preprocess_input(rgb_values, data_format=None):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        rgb_values: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(rgb_values, data_format, mode='torch')


def run():
    model = dense_net_121(weights=None)
    plot_model(model, to_file='densenet121.png', show_shapes=True)
    with open("densenet121.txt", 'wb') as my_file:
        model.summary(print_fn=lambda x: my_file.write(x + '\n'))

if __name__ == "__main__":
    run()
