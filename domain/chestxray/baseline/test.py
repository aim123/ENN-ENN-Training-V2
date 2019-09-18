#!/usr/bin/env python

from keras_applications.resnet50 import ResNet50
from keras_preprocessing import image
from keras_applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights=None)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print("{0} {1}".format(x.shape, x.dtype))
x = np.expand_dims(x, axis=0)
print("{0} {1}".format(x.shape, x.dtype))
x = preprocess_input(x)
print("{0} {1}".format(x.shape, x.dtype))

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print("Predicted: {0}".format(decode_predictions(preds, top=3)[0]))
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
