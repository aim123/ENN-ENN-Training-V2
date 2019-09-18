
from keras.models import Model
from keras.layers import Input, Dense
from keras_applications.densenet import DenseNet121
from keras_applications.densenet import preprocess_input as pi_densenet121
from keras_applications.resnet50 import ResNet50
from keras_applications.resnet50 import preprocess_input as pi_resnet50
from keras_applications.vgg19 import VGG19
from keras_applications.vgg19 import preprocess_input as pi_vgg19


def get_model(img_size, num_channels, num_classes, model_name='densenet121',
              pretrain=False):
    weights = 'imagenet' if pretrain else None
    if pretrain:
        assert img_size == 224
    input_tensor = Input(shape=(img_size, img_size, num_channels))

    if model_name == 'densenet121':
        model = DenseNet121(include_top=False, weights=weights,
                            input_tensor=input_tensor, pooling='avg')
        preprocessor = pi_densenet121

    elif model_name == 'resnet50':
        model = ResNet50(include_top=False, weights=weights,
                         input_tensor=input_tensor, pooling='avg')
        preprocessor = pi_resnet50

    elif model_name == 'vgg19':
        model = VGG19(include_top=False, weights=weights,
                      input_tensor=input_tensor, pooling='avg')
        preprocessor = pi_vgg19

    else:
        raise Exception("Unrecognized model name: %s" % model_name)

    output_tensor = model(input_tensor)
    output_tensor = Dense(num_classes, activation='sigmoid')(output_tensor)
    complete_model = Model(input_tensor, output_tensor)
    return complete_model, preprocessor
