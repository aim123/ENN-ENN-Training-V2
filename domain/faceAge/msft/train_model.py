#
# Training script for reproducing ms ds solution on a single GPU.
#

# Standard imports
import argparse
import glob
import os
import pickle

# Third party imports
import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa

import keras.backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import Sequence
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array

# Custom imports
from domain.chestxray.msft import azure_chestxray_keras_utils
from domain.chestxray.msft.preprocess_labels import preprocess_labels
from domain.chestxray.msft.eval_model import evaluate

ia.seed(1)

class DataGenSequence(Sequence):

    # Tied for Public Enemy #5 for too-many-arguments
    # pylint: disable=too-many-arguments
    def __init__(self, image_dir, batch_size, labels,
                    image_file_index, current_state,
                    resized_height, resized_width, num_channel, num_classes,
                    seq):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.labels = labels
        self.img_file_index = image_file_index
        self.current_state = current_state
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.num_channel = num_channel
        self.num_classes = num_classes
        self.seq = seq

        self.len = len(self.img_file_index) // self.batch_size
        print(("for DataGenSequence", current_state, "total rows are:",
                len(self.img_file_index), ", len is", self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # print("loading data segmentation", idx)
        # Make sure each batch size has the same amount of data
        current_batch = self.img_file_index[idx * self.batch_size: (idx + 1) * self.batch_size]
        input_data = np.empty((self.batch_size, self.resized_height,
                                self.resized_width, self.num_channel))
        target_data = np.empty((self.batch_size, self.num_classes))

        for i, image_name in enumerate(current_batch):
            path = os.path.join(self.image_dir, image_name)
            assert os.path.exists(path)
            # Loading data
            img = load_img(path, target_size=(self.resized_height, self.resized_width))
            img = img_to_array(img)
            input_data[i, :, :, :] = img
            target_data[i, :] = self.labels[image_name]

        # Only do data augmentation in training status
        if self.current_state == 'train':
            x_augmented = self.seq.augment_images(input_data)
        else:
            x_augmented = input_data

        return x_augmented, target_data


# Tied for Public Enemy #9 for too-many-locals
# pylint: disable=too-many-locals
# Public Enemy #10 for too-many-statements
# pylint: disable=too-many-statements
def train(config):
    # Create output directory. To be safe, disallow overwriting previous experiments.
    assert not os.path.exists(config['output_dir'])
    os.makedirs(config['output_dir'])
    config_file = config['output_dir'] + '/config.pkl'

    # Save config for bookkeeping.
    with open(config_file, 'w') as my_file:
        pickle.dump(config, my_file)

    # Training parameters

    initial_lr = config['initial_lr']
    resized_height = config['input_size']
    resized_width = config['input_size']
    image_file_index = config['image_file_index'] # XXX ???
    num_channel = 3
    num_classes = 14
    epochs = config['epochs']

    # Create data augmentation object
    if config['flip_only']:
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
        ], random_order=True)  # apply augmenters in random order
    elif config['augment']:
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Affine(rotate=(-15, 15)),  # random rotate image
            iaa.Affine(scale=(0.8, 1.1)),  # randomly scale the image
        ], random_order=True)  # apply augmenters in random order

    # Location of images for on-the-fly loading.
    # nih_chest_xray_data_dir = config['image_dir']

    # Generator for train and validation data
    # Use the Sequence class per issue https://github.com/keras-team/keras/issues/1638
    batch_size = config['batch_size']  # This was 48 * num_gpu in the multi gpu code.

    # Custom loss function
    def unweighted_binary_crossentropy(y_true, y_pred):
        """
        Args:
            y_true: true labels
            y_pred: predicted labels

        Returns: the sum of binary cross entropy loss across all the classes

        """
        return K.sum(K.binary_crossentropy(y_true, y_pred))

    # Build model
    print("Building model...")
    model = azure_chestxray_keras_utils.build_model(config)

    # Compile model.
    model.compile(optimizer=Adam(lr=initial_lr), loss=unweighted_binary_crossentropy)

    # Set up callbacks.
    print("Using single GPU")
    weights_dir = config['output_dir']
    my_path = os.path.join(weights_dir,
        'azure_chest_xray_14_weights_712split_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.hdf5')
    model_checkpoint = ModelCheckpoint(my_path, monitor='val_loss',
                            save_weights_only=False, save_best_only=True)

    if config['decay']:
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.1,
                                        patience=config['patience'],
                                        verbose=1, min_lr=1e-6)
        callbacks = [model_checkpoint, reduce_lr_on_plateau]
    else:
        callbacks = [model_checkpoint]

    # Load labels for training.
    if config['resplit_data']:
        config['write_to_file'] = False
        labels, partition = preprocess_labels(config)
    else:
        data_partitions_dir = config['labels_dir']
        partition_path = os.path.join(data_partitions_dir, 'partition14_unormalized_cleaned.pickle')
        label_path = os.path.join(data_partitions_dir, 'labels14_unormalized_cleaned.pickle')
        with open(label_path, 'rb') as my_file:
            labels = pickle.load(my_file)
        with open(partition_path, 'rb') as my_file:
            partition = pickle.load(my_file)

    # Train.
    num_workers = 10  # Number of data generation workers
    current_state = 'train'
    generator = DataGenSequence(labels,
                                batch_size,
                                partition['train'],
                                image_file_index,
                                current_state,
                                resized_height,
                                resized_width,
                                num_channel,
                                num_classes,
                                seq)
    current_state = 'validation'
    validation_data = DataGenSequence(labels,
                                batch_size,
                                partition['valid'],
                                image_file_index,
                                current_state,
                                resized_height,
                                resized_width,
                                num_channel,
                                num_classes,
                                seq)
    model.fit_generator(
        generator=generator,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        workers=num_workers,
        validation_data=validation_data
    )

    print("Done Training.")

    if config['evaluate']:
        print("Evaluating")
        # Evaluate best model.
        saved_model_names = glob.glob(config['output_dir'] + '/*hdf5')
        most_recent_saved_model = max(saved_model_names, key=os.path.getctime)
        config['model_file'] = most_recent_saved_model
        config['labels'] = labels
        config['partition'] = partition
        evaluate(config)

        print("Done evaluating.")


def run():
    parser = argparse.ArgumentParser()

    # Options for training.
    parser.add_argument("--gpu_index",
                        default="0",
                        help="which gpu to run on")
    parser.add_argument("--initial_lr",
                        type=float,
                        default=0.001,
                        help="initial learning rate")
    parser.add_argument("--batch_size",
                        type=int,
                        default=48,
                        help="batch size")
    parser.add_argument("--epochs",
                        type=int,
                        default=200,
                        help="number of epochs for training")
    parser.add_argument("--patience",
                        type=int,
                        default=3,
                        help="number of epochs of plateau before reducing lr")
    parser.add_argument("--input_size",
                        type=int,
                        default=224,
                        help="side length in pixels of resized input")
    parser.add_argument("--pretrain",
                        action="store_true",
                        help="initialize with imagenet weights")
    parser.add_argument("--augment",
                        action="store_true",
                        help="use data augmentation")
    parser.add_argument("--preprocess",
                        action='store_true',
                        help="whether to preprocess label input")
    parser.add_argument("--flip_only",
                        action="store_true",
                        help="only augment images by flipping")
    parser.add_argument("--image_model",
                        default="densenet",
                        help="options: [densenet, resnet]")
    parser.add_argument("--freeze",
                        action="store_true",
                        help="whether to freeze pretrained image model")
    parser.add_argument("--decay",
                        action="store_true",
                        help="do custom learning rate decay")
    parser.add_argument("--image_dir",
                        help="directory where training images are located")
    parser.add_argument("--labels_dir",
                        default="labels/",
                        help="directory where training label data is located")
    parser.add_argument("--output_dir",
                        default="weights/",
                        help="directory where training output is saved")

    # Option to evaluate best model.
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="whether to evaluate roc after training")

    # Options for preprocessing labels.
    parser.add_argument("--resplit_data",
                        action="store_true",
                        help="whether to resplit the training data")
    parser.add_argument("--training_data", default="Data_Entry_2017.csv")
    parser.add_argument("--gold_standard", default="BBox_List_2017.csv")
    parser.add_argument("--blacklist",
                        default="blacklist.csv",
                        help="file listing blacklisted images")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="random seed")
    parser.add_argument("--keep_gold_standard",
                        action="store_true",
                        help="whether to train with gold data")

    args = parser.parse_args()

    # Select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index

    # Set up training config
    config = args.__dict__

    # Train.
    train(config)

if __name__ == '__main__':
    run()
