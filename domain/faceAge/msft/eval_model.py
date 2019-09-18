#
# Script for evaluating a trained model.
#

# Standard Imports
import argparse
import os
import pickle

# Third Party Imports
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.utils import Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# Custom imports
from domain.chestxray.msft import azure_chestxray_utils
from domain.chestxray.msft import azure_chestxray_keras_utils
from domain.chestxray.msft.preprocess_labels import preprocess_labels


# Tied for Public Enemy #7 for too-many-statements
# pylint: disable=too-many-statements
def evaluate(config):
    prj_consts = azure_chestxray_utils.ChestXrayConsts()
    pathologies_name_list = prj_consts.DISEASE_LIST

    stanford_result = [
        0.8094, 0.9248, 0.8638, 0.7345, 0.8676, 0.7802, 0.7680,
        0.8887, 0.7901, 0.8878, 0.9371, 0.8047, 0.8062, 0.9164
    ]

    # Evaluation parameters.
    resized_height = config['input_size']
    resized_width = config['input_size']
    num_channel = 3
    num_classes = 14
    batch_size = 48  # 512

    # Load labels for evaluation.
    if 'labels' in config and 'partition' in config:
        labels = config['labels']
        partition = config['partition']
    elif config['resplit_data']:
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

    # XXX: This generator is duplicated from train_model.py. To be deduplicated.

    # Location of images for on-the-fly loading.
    nih_chest_xray_data_dir = config['image_dir']

    # generator for train and validation data
    # use the Sequence class per issue https://github.com/keras-team/keras/issues/1638
    class DataGenSequence(Sequence):
        def __init__(self, labels, image_file_index, current_state):
            self.batch_size = batch_size
            self.labels = labels
            self.img_file_index = image_file_index
            self.current_state = current_state
            self.len = len(self.img_file_index) // self.batch_size
            print((
                "for DataGenSequence", current_state, "total rows are:",
                len(self.img_file_index), ", len is",
                self.len))

        def __len__(self):
            return self.len

        def __getitem__(self, idx):
            # print("loading data segmentation", idx)
            # make sure each batch size has the same amount of data
            current_batch = self.img_file_index[idx * self.batch_size: (idx + 1) * self.batch_size]
            input_data = np.empty((self.batch_size, resized_height, resized_width, num_channel))
            target_data = np.empty((self.batch_size, num_classes))

            for i, image_name in enumerate(current_batch):
                path = os.path.join(nih_chest_xray_data_dir, image_name)
                if not os.path.exists(path):
                    path = path[:-4] + '.jpg'
                assert os.path.exists(path)

                # loading data

                img = load_img(path, target_size=(resized_height, resized_width))
                img = img_to_array(img)
                input_data[i, :, :, :] = img
                target_data[i, :] = labels[image_name]

                # only do random flipping in training status
            if self.current_state == 'train':
                # this is different from the training code
                x_augmented = input_data
            else:
                x_augmented = input_data

            if config['preprocess']:
                x_augmented = azure_chestxray_keras_utils.preprocess_input(x_augmented, config)

            return x_augmented, target_data

    # Load test images.
    test_len = len(partition['test'])
    #input_data_test = np.empty((test_len, 224, 224, 3), dtype=np.float32)
    target_data_test = np.empty((test_len - test_len % batch_size, 14), dtype=np.float32)

    for i, npy in enumerate(partition['test']):
        if i < len(target_data_test):
            # round to batch_size
            target_data_test[i, :] = labels[npy]

    print(("len of result is", len(target_data_test)))

    # Load model.
    model_file = config['model_file']
    print(("Loading", model_file))
    model = azure_chestxray_keras_utils.build_model(config)
    model.load_weights(model_file)

    # Evaluate model.
    print(("Evaluating model", model_file))
    target_data_pred = model.predict_generator(generator=DataGenSequence(labels,
                                partition['test'], current_state='test'),
                                workers=10, verbose=1, max_queue_size=1)
    print(("result shape", target_data_pred.shape))

    # add one fake row of ones in both test and pred values to avoid:
    # ValueError: Only one class present in target_data_true.
    # ROC AUC score is not defined in that case.
    target_data_test = np.insert(target_data_test, 0, np.ones((target_data_test.shape[1],)), 0)
    target_data_pred = np.insert(target_data_pred, 0, np.ones((target_data_pred.shape[1],)), 0)

    data_frame = pd.DataFrame(columns=['Disease', 'Our AUC Score', 'Stanford AUC Score'])
    for frame in range(14):
        data_frame.loc[frame] = [pathologies_name_list[frame],
                     roc_auc_score(target_data_test[:, frame], target_data_pred[:, frame]),
                     stanford_result[frame]]

    data_frame['Delta'] = data_frame['Stanford AUC Score'] - data_frame['Our AUC Score']
    data_frame.to_csv(model_file + ".csv", index=False)
    print(data_frame)
    print((data_frame.mean()))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_index",
                        help="which gpu to run on")
    parser.add_argument("--input_size",
                        type=int,
                        default=224,
                        help="side length in pixels of resized input")
    parser.add_argument("--image_model",
                        default="densenet",
                        help="options: [densenet, resnet]")
    parser.add_argument("--image_dir",
                        help="directory where training images are located")
    parser.add_argument("--labels_dir",
                        default="labels/",
                        help="directory where training label data is located")
    parser.add_argument("--model_file",
                        help="path to saved model weights")
    parser.add_argument("--preprocess",
                        action='store_true',
                        help="whether to preprocess label input")

    # Options for preprocessing labels.
    parser.add_argument("--resplit_data",
                        action="store_true",
                        help="whether to resplit the training data")
    parser.add_argument("--training_data",
                        default="Data_Entry_2017.csv")
    parser.add_argument("--gold_standard",
                        default="BBox_List_2017.csv")
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
    evaluate(config)

if __name__ == '__main__':
    run()
