#
# Script for preprocessing and splitting label data for the chestxray dataset.
#
# Requires Data_Entry_2017.csv and BBox_List_2017.csv, which are provided with the dataset.
#
# Important note: this splits data differently for different versions of python random.
#

# Standard imports
import argparse
import pickle
import random
import re

# Third party imports
import pandas as pd
import numpy as np
import tqdm

# Custom imports
from domain.chestxray.msft import azure_chestxray_utils



# Public Enemy #9 for too-many-statements
# pylint: disable=too-many-statements
def preprocess_labels(config):
    # Data file paths.
    nih_patients_and_labels_file = config['training_data']
    nih_annotated_file = config['gold_standard']

    # Load automatically-annotated data
    labels_df = pd.read_csv(nih_patients_and_labels_file)

    # Load gold standard radiologist annotated data to exclude from training.
    bbox_df = pd.read_csv(nih_annotated_file)
    bbox_patient_index_df = bbox_df['Image Index'].str.slice(3, 8).astype(int)

    # Load any blacklisted image names.
    ignored_images_set = set()
    if config['blacklist'] is not None:
        with open(config['blacklist'], 'r') as my_file:
            for line in my_file:
                # delete the last char which is \n
                ignored_images_set.add(line[:-1])
                if int(line[:-9]) >= 30805:
                    print((line[:-1]))

    # Remove gold standard data from training set.
    all_patient_ids = set(labels_df['Patient ID'])
    bbox_patient_ids = set(bbox_patient_index_df.values)

    # Either keep or remove gold data from train.
    if config['keep_gold_standard']:
        patient_ids = all_patient_ids
    else:
        patient_ids = all_patient_ids - bbox_patient_ids
    patient_ids = list(patient_ids)

    print(("Total number of patient ids", len(all_patient_ids)))
    print(("Number of unique gold standard patient ids", len(bbox_patient_ids)))
    print(("Number of cleaned patient ids for training", len(patient_ids)))
    print(("Total number of gold standard patient ids", bbox_df.shape[0]))

    # Split data into train, val, and test.
    random.seed(config['seed'])  # The effect of this may depend on python version.
    random.shuffle(patient_ids)

    # Following the MSFT example, we do a 7:1:2 split.
    total_patient_number = len(all_patient_ids)  # Note that the uncleaned lenghth is used here.
    patient_id_train = patient_ids[:int(total_patient_number * 0.7)]
    patient_id_valid = patient_ids[int(total_patient_number * 0.7):int(total_patient_number * 0.8)]
    patient_id_test = patient_ids[int(total_patient_number * 0.8):]

    # Add gold standard to test if they have been withheld from train.
    if not config['keep_gold_standard']:
        patient_id_test.extend(list(bbox_patient_ids))
        patient_id_test = list(set(patient_id_test))

    print(("train:{} valid:{} test:{}".format(
        len(patient_id_train), len(patient_id_valid), len(patient_id_test))))
    prj_consts = azure_chestxray_utils.ChestXrayConsts()

    # Function for grabbing labels from df.
    def process_data(current_df, patient_ids):
        pathologies_name_list = prj_consts.DISEASE_LIST
        image_name_index = []
        image_labels = {}
        for individual_patient in tqdm.tqdm(patient_ids):
            for _, row in current_df[current_df['Patient ID'] == individual_patient].iterrows():
                processed_image_name = row['Image Index']
                if processed_image_name not in ignored_images_set:
                    image_name_index.append(processed_image_name)
                    image_labels[processed_image_name] = np.zeros(14, dtype=np.uint8)
                    for disease_index, ele in enumerate(pathologies_name_list):
                        # Added replace below to handle 'Pleural Thickening'
                        # vs 'Pleural_Thickening' bug.
                        if re.search(ele.replace(' ', '_'), row['Finding Labels'], re.IGNORECASE):
                            image_labels[processed_image_name][disease_index] = 1
        return image_name_index, image_labels

    # # create and save train/test/validation partitions list

    train_data_index, train_labels = process_data(labels_df, patient_id_train)
    valid_data_index, valid_labels = process_data(labels_df, patient_id_valid)
    test_data_index, test_labels = process_data(labels_df, patient_id_test)

    print(("train, valid, test image number is:",
          len(train_data_index), len(valid_data_index), len(test_data_index)))

    # save the data
    labels_all = {}
    labels_all.update(train_labels)
    labels_all.update(valid_labels)
    labels_all.update(test_labels)

    partition_dict = {'train': train_data_index,
                      'test': test_data_index,
                      'valid': valid_data_index}

    if config['write_to_file']:
        with open('labels14_unormalized_cleaned.pickle', 'wb') as my_file:
            pickle.dump(labels_all, my_file)

        with open('partition14_unormalized_cleaned.pickle', 'wb') as my_file:
            pickle.dump(partition_dict, my_file)

        # also save the patient id partitions for pytorch training
        with open('train_test_valid_data_partitions.pickle', 'wb') as my_file:
            pickle.dump([patient_id_train, patient_id_valid,
                         patient_id_test, bbox_patient_ids], my_file)

    return labels_all, partition_dict


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int,
                        default=0,
                        help="random seed")
    parser.add_argument("--blacklist",
                        default=None,
                        help="file listing blacklisted images")
    parser.add_argument("--training_data",
                        default="Data_Entry_2017.csv")
    parser.add_argument("--gold_standard",
                        default="BBox_List_2017.csv")
    parser.add_argument("--write_to_file",
                        action="store_true",
                        help="whether to write labels to file")
    parser.add_argument("--keep_gold_standard",
                        action="store_true",
                        help="whether to train with gold data")
    args = parser.parse_args()

    config = args.__dict__

    preprocess_labels(config)

    print("Done.")

if __name__ == '__main__':
    run()
