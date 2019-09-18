### Copyright (C) Microsoft Corporation.

#
# Copied from microsoft codebase, and updated to include ResNet50.
#
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras_applications.resnet50 import ResNet50
from keras_applications.densenet import DenseNet121
from keras_applications.densenet import DenseNetImageNet121
from keras_applications.densenet import DenseNetImageNet201
from keras_applications.imagenet_utils import preprocess_input as keras_preprocess_input


# Model building function
def build_model(config):
    """
    Returns: a model with specified weights
    """

    input_size = config['input_size']
    input_shape = (input_size, input_size, 3)

    if ('pretrain' in config) and config['pretrain']:
        assert config['input_size'] == 224
        weights = 'imagenet'
    else:
        weights = None

    assert config['image_model'] in ['densenet', 'resnet', 'linear']
    if config['image_model'] == 'densenet':
        print("Using Densenet.")
        base_model = DenseNet121(input_shape=input_shape,
                                 weights=weights,
                                 include_top=False,
                                 pooling='avg')
        image_input = base_model.input
        layer = base_model.output
    elif config['image_model'] == 'resnet':
        print("Using Resnet.")
        base_model = ResNet50(input_shape=input_shape,
                              weights=weights,
                              include_top=False,
                              pooling='avg')
        image_input = base_model.input
        layer = base_model.output
    elif config['image_model'] == 'linear':
        print("Using linear model.")
        image_input = Input(shape=input_shape)
        layer = Flatten()(image_input)

    if ('freeze' in config) and config['freeze']:
        for layer in base_model.layers:
            try:
                layer.trainable = False
                print("Freezing {}".format(layer))
            except Exception:
                print("Not trainable {}".format(layer))

    predictions = Dense(14, activation='sigmoid')(layer)
    model = Model(inputs=image_input, outputs=predictions)
    return model


# Input preprocessing function.
def preprocess_input(layer, config):
    assert config['image_model'] in ['densenet', 'resnet', 'linear']
    if config['image_model'] == 'densenet':
        mode = 'torch'
    elif config['image_model'] == 'resnet':
        mode = 'caffe'
    elif config['image_model'] == 'linear':
        mode = 'tf'
    return keras_preprocess_input(layer, mode=mode)


def run():
    model = build_model(DenseNetImageNet121)
    print(model.summary())
    model = build_model(DenseNetImageNet201)
    print(model.summary())

if __name__ == "__main__":
    run()
