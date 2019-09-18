#!/usr/bin/env python

from __future__ import print_function

import os
import random

from future.builtins import range

import numpy as np
import h5py

from sklearn.metrics import roc_auc_score

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import generic_utils

from model_zoo import get_model

# Path to data file for x-ray dataset
DATA_FILE = 'output/chest_xray_224.h5'
# Directory to output all of the results
OUTDIR = 'train_224_pretrain'
# The names of the classes
LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
          'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
          'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

# Test split for data
TEST_SPLIT = 0.2
# Val split for data
VAL_SPLIT = 0.1
# Random seed for deterministic splits
SEED = 1337
# Use pretrained weights or not
PRETRAIN = True
# Name of model to use (currently: densenet121, resnet50, vgg19)
MODEL_NAME = 'densenet121'

# Skip training and go directly to testing
TEST_ONLY = False
# Number of epochs of training
EPOCHS_TRAINING = 30
# Batch size for training
BATCH_SIZE = 16
# Init learning rate
LR_INIT = 0.001
# Learning rate decay rate
LR_DECAY = 0.1
# LR decay interval
LR_DECAY_INTEVAL = 10
# Use 'auto' for model's preprocessing method, 'standard' for mean/var norm,
# 'simple' for divide by 255, and None for no preprocessing
PREPROCESSING = 'auto'
# Fudge factor for preprocessing
EPSILON = 1e-12
# Checkpoint every X number of epochs
CHECKPOINT_INTERVAL = 1
# Name of checkpoint file to load, None to not load any checkpoints
# CHECKPOINT_FILE = os.path.join(OUTDIR, "checkpoint_E9_F0.5006621173289034.h5")
CHECKPOINT_FILE = None

def init():
    np.set_printoptions(threshold='nan')
    config = tf.ConfigProto()
    # XXX gpu_options is obsolete
    # config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))


def load_data():
    # Sanity check to make sure labels are in alphabetical order
    assert LABELS == sorted(LABELS)
    assert VAL_SPLIT >= 0 and TEST_SPLIT >= 0 and VAL_SPLIT + TEST_SPLIT <= 1.0
    random.seed(SEED)
    np.random.seed(SEED)

    print("Opening data file {}".format(DATA_FILE))
    with h5py.File(DATA_FILE, 'r') as h5_data:
        data = np.array(h5_data.get('images'))
        print("Loaded images: {}".format(str(data.shape())))
        hstack_input = []
        for label in LABELS:
            my_h5_data = h5_data.get(label)
            my_np_array = np.array(my_h5_data)
            my_reshape = np.reshape(my_np_array, (-1, 1))
            hstack_input.append(my_reshape)
        labels = np.hstack(hstack_input)
        print("Loaded labels: {}".format(str(labels.shape())))

    num_samples = len(data)
    img_size = data.shape[1]
    data = data.astype(np.float32, copy=False)
    if PRETRAIN:
        new_data = np.empty((num_samples, img_size, img_size, 3), dtype=np.float32)
        new_data[:, :, :, 0] = data[:, :, :, 0]
        data = new_data
    labels = labels.astype(np.float32, copy=False)
    print("Formatted data/labels to float32")
    # print labels

    train_split = 1.0 - TEST_SPLIT - VAL_SPLIT
    n_train = int(train_split * num_samples)
    n_test = int(TEST_SPLIT * num_samples)
    n_val = num_samples - n_train - n_test

    indices = np.random.permutation(num_samples)
    train_idx = indices[:n_train]
    indices = indices[n_train:]
    test_idx = indices[:n_test]
    val_idx = indices[n_test:]
    print("Split data into {0} train, {1} test, and {2} val".format(
                n_train, n_test, n_val))

    return data[train_idx], labels[train_idx], \
           data[val_idx], labels[val_idx], data[test_idx], labels[test_idx]


def shuffle(seq):
    seed = random.randint(1, 10e6)
    for data in seq:
        np.random.seed(seed)
        np.random.shuffle(data)


def calculate_class_weights(labels):
    class_weights = np.sum(labels, axis=0)
    no_finding = np.count_nonzero(np.sum(labels, axis=1))
    class_weights = np.concatenate([np.array([no_finding]), class_weights])
    class_weights = class_weights.astype(np.float32)
    class_weights = 1. / (class_weights / np.max(class_weights))

    print("Class weights: {}".format(class_weights))
    print("Class weights as dict: {}".format(
            dict(list(zip(['No_Finding'] + LABELS, class_weights)))))
    return class_weights


def augment(input_data):
    def flip_axis(inlist, axis):
        inlist = np.asarray(inlist).swapaxes(axis, 0)
        inlist = inlist[::-1, ...]
        inlist = inlist.swapaxes(0, axis)
        return inlist

    for i in range(input_data.shape[0]):
        if np.random.random() < 0.5:
            input_data[i] = flip_axis(input_data[i], 1)
    return input_data


def get_data_gen(input_data, target_data, class_weights=None):
    for i in range(0, input_data.shape[0], BATCH_SIZE):
        input_batch = input_data[i:i + BATCH_SIZE]
        target_batch = target_data[i:i + BATCH_SIZE]

        if input_batch.shape[0] != BATCH_SIZE:
            return

        if class_weights is not None:
            weights = np.empty(BATCH_SIZE)
            for j in range(BATCH_SIZE):
                if np.sum(target_batch[j]) == 0:
                    weights[j] = class_weights[0]
                else:
                    weights[j] = np.dot(class_weights[1:], target_batch[j])
            yield augment(input_batch), target_batch, weights
        else:
            yield input_batch, target_batch


def learning_rate_function(learning_rate, epoch):
    new_lr = LR_DECAY ** (epoch / LR_DECAY_INTEVAL) * learning_rate
    print("Current learning rate: {}".format(new_lr))
    return new_lr


def test(model, input_data, target_data):
    generator = get_data_gen(input_data, target_data)
    all_preds = []
    progbar = generic_utils.Progbar(input_data.shape[0], interval=0.0)

    # Note: _ is pythonic for an unused variable: Was target_batch
    for input_batch, _ in generator:
        all_preds.append(model.predict_on_batch(input_batch))
        progbar.add(BATCH_SIZE)
    print()
    pred_target = np.vstack(all_preds)
    target_data = target_data[:len(pred_target)]
    try:
        scores = roc_auc_score(target_data, pred_target, average=None)
    except Exception:
        scores = np.zeros(len(LABELS))
    print("AUC Scores: {}".format(dict(list(zip(LABELS, scores)))))
    print("Mean: {}".format(np.mean(scores)))
    return scores


# Tied for Public Enemy #5 for too-many-locals
# pylint: disable=too-many-locals
# Public Enemy #6 for too-many-statements
# pylint: disable=too-many-statements
def train():
    train_input, train_target, val_input, val_target, test_input, test_target = load_data()

    img_size = train_input.shape[1]
    num_channels = train_input.shape[-1]
    num_classes = train_target.shape[1]
    num_samples = train_input.shape[0]

    if PRETRAIN:
        assert PREPROCESSING == 'auto'
    model, preprocess_input = get_model(img_size, num_channels, num_classes,
                                        model_name=MODEL_NAME, pretrain=PRETRAIN)

    if CHECKPOINT_FILE is not None:
        assert os.path.exists(CHECKPOINT_FILE)
        model.load_weights(CHECKPOINT_FILE)
        epoch = int(CHECKPOINT_FILE.split('_')[-2][1:]) + 1
        lr_init = learning_rate_function(LR_INIT, epoch)
        print("Loaded weights from {}".format(CHECKPOINT_FILE))
    else:
        epoch = 0
        lr_init = LR_INIT
    shape_str = "({0}, {0}, {1})".format(img_size, num_channels)
    print("Finished loading {0} with {1} input shape (pretrained: {2})".format(
                                MODEL_NAME, shape_str, PRETRAIN))

    if PREPROCESSING == 'auto':
        train_input = preprocess_input(train_input)
        test_input = preprocess_input(test_input)
        val_input = preprocess_input(val_input)
    elif PREPROCESSING == 'standard':
        row_axis = 1
        col_axis = 2
        channel_axis = 3
        mean = np.mean(train_input, axis=(0, row_axis, col_axis))
        broadcast_shape = [1, 1, 1]
        broadcast_shape[channel_axis - 1] = train_input.shape[channel_axis]
        mean = np.reshape(mean, broadcast_shape)
        # x -= mean
        train_input -= mean
        test_input -= mean
        val_input -= mean
        std = np.std(train_input, axis=(0, row_axis, col_axis))
        broadcast_shape = [1, 1, 1]
        broadcast_shape[channel_axis - 1] = train_input.shape[channel_axis]
        std = np.reshape(std, broadcast_shape)
        # x /= (std + K.epsilon())
        train_input /= (std + EPSILON)
        test_input /= (std + EPSILON)
        val_input /= (std + EPSILON)
    elif PREPROCESSING == 'simple':
        train_input /= 255.
        test_input /= 255.
        val_input /= 255
    print("Finished preprocessing data using method: {}".format(PREPROCESSING))

    optimizer = Adam(lr=lr_init, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    print("Constructed and compiled model")

    if not TEST_ONLY:
        ###### Start training ######
        # Note: _ is pythonic for an unused variable: Was i
        for _ in range(EPOCHS_TRAINING):
            print()
            print("Starting training for epoch {}".format(epoch))
            shuffle([train_input, train_target])
            generator = get_data_gen(train_input, train_target, class_weights=None)
            K.set_value(model.optimizer.lr, learning_rate_function(LR_INIT, epoch))

            progbar = generic_utils.Progbar(num_samples, interval=0.0)
            for input_batch, target_batch in generator:
                loss = model.train_on_batch(input_batch, target_batch)
                progbar.add(BATCH_SIZE, values=[("train_loss", loss)])
            print()

            val_auc_scores = test(model, val_input, val_target)
            mean_score = np.mean(val_auc_scores)
            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                model_file = os.path.join(OUTDIR, "checkpoint_E%s_F%s.h5" % \
                                          (epoch, round(mean_score, 3)))
                model.save_weights(model_file)
                print("Saved weights to {}".format(model_file))

            epoch += 1
        ###### Finished training ######

    print("Test results:")
    test_auc_scores = test(model, test_input, test_target)
    print(test_auc_scores)

if __name__ == "__main__":
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    init()
    train()
