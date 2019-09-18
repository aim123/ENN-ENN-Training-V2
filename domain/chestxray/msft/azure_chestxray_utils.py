### Copyright (C) Microsoft Corporation.

import os
import numpy as np


class ChestXrayConsts():
    # Number of classes for each disease
    DISEASE_NUM_CLASSES = {
        'Atelectasis': 2,
        'Cardiomegaly': 2,
        'Consolidation': 2,
        'Edema': 2,
        'Effusion': 2,
        'Emphysema': 2,
        'Fibrosis': 2,
        'Hernia': 2,
        'Infiltration': 2,
        'Mass': 2,
        'Nodule': 2,
        'Pleural Thickening': 2,
        'Pneumonia': 2,
        'Pneumothorax': 2
    }

    DISEASE_LIST = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                    'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']

    PRETRAINED_DENSENET201_IMAGENET_CHESTXRAY_MODEL_FILE_NAME = \
    'chexnet_14_weights_multigpu_contribmodel_121layer_712split_epoch_011_val_loss_153.9783.hdf5'
    FULLY_PRETRAINED_MODEL_DIR_list = ['fully_trained_models']

    CHESTXRAY_MODEL_EXPECTED_IMAGE_HEIGHT = 224
    CHESTXRAY_MODEL_EXPECTED_IMAGE_WIDTH = 224

    BASE_INPUT_DIR_list = ['chestxray', 'data', 'ChestX-ray8']
    BASE_OUTPUT_DIR_list = ['chestxray', 'output']
    CREDENTIALS_DIR_list = ['code', 'notShared']

    SRC_DIR_list = ['Code', 'src']
    ChestXray_IMAGES_DIR_list = ['ChestXray-NIHCC']
    ChestXray_OTHER_DATA_DIR_list = ['ChestXray-NIHCC_other']
    PROCESSED_IMAGES_DIR_list = ['processed_npy14']
    DATA_PARTITIONS_DIR_list = ['data_partitions']
    MODEL_WEIGHTS_DIR_list = ['weights_tmpdir']

    def __setattr__(self, *_):
        raise TypeError


# os agnostic 'ls' function
def get_files_in_dir(crt_dir):
    return ([f for f in os.listdir(crt_dir) if os.path.isfile(os.path.join(crt_dir, f))])


def normalize_nd_array(crt_array):
    # Normalised [0,1]
    crt_array = crt_array - np.min(crt_array)
    return crt_array / np.ptp(crt_array)


def print_image_stats_by_channel(crt_image):
    print('min:')
    print((np.amin(crt_image[:, :, 0]),
          np.amin(crt_image[:, :, 1]),
          np.amin(crt_image[:, :, 2])))
    print('max:')
    print((np.amax(crt_image[:, :, 0]),
          np.amax(crt_image[:, :, 1]),
          np.amax(crt_image[:, :, 2])))


if __name__ == "__main__":
    PRJ_CONSTS = ChestXrayConsts()
    print(('model_expected_image_height = ', PRJ_CONSTS.CHESTXRAY_MODEL_EXPECTED_IMAGE_HEIGHT))
    print(('model_expected_image_width = ', PRJ_CONSTS.CHESTXRAY_MODEL_EXPECTED_IMAGE_WIDTH))
