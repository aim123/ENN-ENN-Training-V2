### Copyright (C) Microsoft Corporation.

#
# Copied from microsoft codebase, and updated to include ResNet50.
#

import os

from future.builtins import range

import numpy as np

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.utils import Sequence
from keras.utils import to_categorical

import imgaug as ia
from imgaug import augmenters as iaa

from domain.faceAge.keras_densenet import preprocess_input


class DataGenSequence(Sequence):

    # Public Enemy #2 for too-many-instance-attributes
    # pylint: disable=too-many-instance-attributes
    def __init__(self, train_config, train_info, train_model, model_evaluator,
                 image_dir, labels, img_file_index, current_state):

        self.train_info = train_info
        self.train_model = train_model
        self.model_evaluator = model_evaluator

        self.image_dir = image_dir
        self.labels = labels
        self.img_file_index = img_file_index
        self.current_state = current_state

        self.batch_size = train_config.get('batch_size')
        self.len = len(self.img_file_index) // self.batch_size
        self.constant_input = np.zeros(self.batch_size)

        self.resized_height = self.resized_width = train_info.get('image_size')
        self.num_channel = train_info.get('num_channels')
        
        #$$$$$$$$$$$$$$$ change start $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #change definition of num_classes 
        #self.num_classes = len(train_info.get('diseases'))
        self.num_classes = train_info.get('num_classes')
        
        #$$$$$$$$$$$$$$$ change end $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        self.augment = train_info.get('augment')
        self.test_augment = train_info.get('test_augment')
        self.encoder = train_info.get('encoder')
        self.encoder_weights = train_info.get('encoder_weights')
        self.multitask = train_info.get('multitask')
        self.num_tasks = train_info.get('num_tasks')
        
        #$$$$$$$$$$$$$$$ change start $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # comment below variable, as it is not needed
        #self.disease_sizes = train_info.get('disease_sizes')
        
        #$$$$$$$$$$$$$$$ change end $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("self.num_tasks in datagen",self.num_tasks)
        
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("self.multitask  in datagen",self.multitask)
        
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("self.num_classes  in datagen",self.num_classes)
        
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("self.num_channel  in datagen",self.num_channel)

        self.init_externals()

        # Create data augmentation object
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Affine(rotate=(-15, 15)),  # random rotate image
            iaa.Affine(scale=(0.8, 1.1)),  # randomly scale the image
        ], random_order=True)  # apply augmenters in random order

        print("For DataGenSequence {0}, total rows are: {1}, length is {2}".format(
            current_state, len(self.img_file_index), self.len))

    def __len__(self):
        return self.len

    def _convert_to_onehot(self, labels, num):
        assert len(labels.shape) == 2
        return to_categorical(labels, num)

    def __getitem__(self, idx):

        # Make sure each batch size has the same amount of data
        # print "loading data segmentation", idx
        current_batch = self.img_file_index[idx * self.batch_size: (idx + 1) * \
                                                               self.batch_size]
        input_data = np.empty((self.batch_size, self.resized_height, self.resized_width,
                      self.num_channel))
        target_data = np.empty((self.batch_size, self.num_classes))
        
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("target_data",target_data)
        
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("target_data",target_data.shape)

        for i, image_name in enumerate(current_batch):
            path = os.path.join(self.image_dir, image_name)
            assert os.path.exists(path)

            # Loading image to array
            img = load_img(path, target_size=(self.resized_height,
                                              self.resized_width))
            img = img_to_array(img)

            input_data[i, :, :, :] = img
            target_data[i, :] = self.labels[image_name]

        # Only do data augmentation in training status
        if (self.current_state == 'train' and self.augment) or \
            (self.current_state != 'train' and self.test_augment):
            x_augmented = self.seq.augment_images(input_data)
        else:
            x_augmented = input_data

        # Normalize between 0 and 1
        if self.encoder is not None and \
            self.encoder_weights == 'imagenet':
            x_augmented = preprocess_input(x_augmented)
            assert np.isfinite(x_augmented).all()
        else:
            x_augmented /= 255.

        if self.multitask:
            if self.encoder is None:
                input_data_batch = [x_augmented for i in range(self.num_tasks)]
            else:
                input_data_batch = [x_augmented]

            target_data_batch = []
            for i in range(self.num_tasks):
                disease_size = self.disease_sizes[i]
                target_data_batch.append(self._convert_to_onehot(
                                                target_data[:, i:i + 1],
                                                disease_size))
        else:
            input_data_batch = [x_augmented]
            target_data_batch = [target_data]

        input_data_batch = self.model_evaluator.determine_train_data_inputs(
            self.train_model, input_data_batch)
            
        
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("input_data_batch in datagen",input_data_batch)
        
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("target_data_batch in datagen",target_data_batch)
        
        return input_data_batch, target_data_batch

    def init_externals(self):
        np.set_printoptions(suppress=True)
        ia.seed(1)
