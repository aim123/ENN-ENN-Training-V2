#
# Util for loading omniglot data.
#
import glob
import json
import os
import random
import shutil
import string

import numpy as np

from imageio import imread
from keras.utils import to_categorical
from domain.omniglot.random_alphabet_ordering import get_random_alphabet_ordering

NUM_EX_PER_CHARACTER = 20
IMG_SRC_FILES = glob.glob('/home/jason/Desktop/omniglot/python/images_background/*/*/*') + \
                glob.glob('/home/jason/Desktop/omniglot/python/images_evaluation/*/*/*')


def rand_string(string_len):
    return ''.join(random.choice(string.ascii_lowercase + \
                                 string.ascii_uppercase + \
                                 string.digits) for _ in range(string_len))


def generate_split(train, val, test):
    np.random.seed(1337)
    my_set = set(np.arange(NUM_EX_PER_CHARACTER))
    train_id = set(np.random.choice(list(my_set), train, replace=False))
    my_set = my_set - train_id
    val_id = set(np.random.choice(list(my_set), val, replace=False))
    my_set = my_set - val_id
    assert len(my_set) == test
    test_id = my_set

    return train_id, val_id, test_id

    # print "Train Id:", train_id
    # print "Val Id:", val_id
    # print "Test Id:", test_id


def separate_images(train=10, val=4, test=6):
    if not os.path.exists("train"):
        os.makedirs("train")
    if not os.path.exists("val"):
        os.makedirs("val")
    if not os.path.exists("test"):
        os.makedirs("test")

    split_dict = {}
    for i, image_path in enumerate(IMG_SRC_FILES):
        alphabet, character, image = image_path.split('/')[-3:]
        alphabet = alphabet.replace("_", "-")
        print("{0} {1} {2}".format(alphabet, character, image))

        if (alphabet, character) not in split_dict:
            split_dict[(alphabet, character)] = generate_split(train, val, test)
        train_id, val_id, test_id = split_dict[(alphabet, character)]

        img_num = int(image.split(".")[0].split("_")[1]) - 1
        new_image_name = "%s_%s_%s.png" % (alphabet, character, rand_string(16))
        print("{0} {1}".format(i, image_path))
        if img_num in train_id:
            new_image_name = "train/" + new_image_name
        elif img_num in val_id:
            new_image_name = "val/" + new_image_name
        else:
            assert img_num in test_id
            new_image_name = "test/" + new_image_name
        shutil.copyfile(image_path, new_image_name)


def get_alphabet_size():
    data_dict = load_new_dataset()
    train_dict = data_dict['Y_train']
    size_dict = {}
    for alphabet in train_dict:
        size_dict[alphabet] = train_dict[alphabet].shape[1]
    with open("alphabet_info.json", 'wb') as my_file:
        json.dump(size_dict, my_file)


def load_new_dataset(data_file=None, alphabets=None, number_tasks=None,
                     unzip_curr_dir=False):

    assert os.path.exists(data_file)

    if unzip_curr_dir:
        data_file_base_dir = "omniglot"
    else:
        data_file_base_dir = os.path.join(os.path.dirname(data_file),
                                          "omniglot")

    if not os.path.exists(data_file_base_dir):
        os.makedirs(data_file_base_dir)
    if not os.path.exists(os.path.join(data_file_base_dir, "train")) or \
            not os.path.exists(os.path.join(data_file_base_dir, "val")) or \
            not os.path.exists(os.path.join(data_file_base_dir, "test")):

        tar_command = "tar -xvzf {0} -C {1}".format(data_file,
                                                    data_file_base_dir)
        os.system(tar_command)

    if alphabets is not None:
        task_alphabets = alphabets
    else:
        task_alphabets = [x.replace("_", "-") for x in \
                          get_random_alphabet_ordering()]
        if number_tasks is not None:
            task_alphabets = task_alphabets[:number_tasks]
    print("loading following alphabets: {}".format(task_alphabets))

    data_dict = {}
    data_dict["X_train"], data_dict["Y_train"] = load_helper(
        os.path.join(data_file_base_dir, "train"), task_alphabets)
    data_dict["X_test"], data_dict["Y_test"] = load_helper(
        os.path.join(data_file_base_dir, "test"), task_alphabets)
    data_dict["X_val"], data_dict["Y_val"] = load_helper(
        os.path.join(data_file_base_dir, "val"), task_alphabets)

    return data_dict


def load_helper(directory, alphabets):
    my_input = {}
    my_output = {}
    for img_file in glob.glob(directory + "/*.png"):
        # print img_file

        # Note: _ is pythonic for an unused variable. Was: image
        alphabet, character, _ = img_file.split('/')[-1].split(".")[0].split("_")
        if alphabet not in alphabets:
            continue
        if alphabet not in my_input:
            my_input[alphabet] = []
            my_output[alphabet] = []
        img = imread(img_file)
        img = np.reshape(img, (105, 105, 1))
        img = - ((img / 255.0) - 1.0)
        label = int(character[-2:]) - 1
        my_input[alphabet].append(img)
        my_output[alphabet].append(label)

    for alphabet in my_input:
        my_input[alphabet] = np.array(my_input[alphabet])
        my_output[alphabet] = to_categorical(my_output[alphabet])
    return my_input, my_output


# Legacy functions
def download_and_unpack(data_file=None):

    assert os.path.exists(data_file)

    data_file_base_dir = os.path.dirname(data_file)
    if not os.path.exists(os.path.join(data_file_base_dir,
                                       'images_evaluation')) or not os.path.exists(
        os.path.join(data_file_base_dir, 'images_background')):
        tar_command = "tar -xvzf {0} -C {1}".format(data_file,
                                                    data_file_base_dir)
        os.system(tar_command)

    img_src_files = glob.glob(
        os.path.join(data_file_base_dir, 'images_evaluation/*/*/*')) + \
                    glob.glob(os.path.join(data_file_base_dir, 'images_background/*/*/*'))
    return img_src_files


# Tied for Public Enemy #10 for too-many-locals
# pylint: disable=too-many-locals
def load_tasks(pad_input=False, train_samples_per_character=16,
               num_filters=12, num_tasks=50, data_file=None):
    tasks = {}
    task_alphabets = get_random_alphabet_ordering()[:num_tasks]
    img_src_files = download_and_unpack(data_file)

    print("Loading {} images...".format(len(img_src_files)))
    for img_src_file in img_src_files:
        # Note: _ is pythonic for an unused variable. Was: image
        alphabet, character, _ = img_src_file.split('/')[-3:]
        if alphabet in task_alphabets:
            if alphabet not in tasks:
                tasks[alphabet] = {'X': [], 'Y': []}
            task = tasks[alphabet]
            one_input = imread(img_src_file)
            one_input = np.reshape(one_input, (105, 105, 1))
            one_output = int(character[-2:]) - 1
            task['X'].append(one_input)
            task['Y'].append(one_output)

    print("Preprocessing task data...")
    for alphabet in tasks:
        print(alphabet)
        task = tasks[alphabet]

        # Shuffle and normalize data.
        shuffled = np.arange(len(task['X']))
        np.random.shuffle(shuffled)
        input_arr = - ((np.array(task['X'])[shuffled] / 255) - 1)
        target_arr = np.array(task['Y'])[shuffled]
        if pad_input:
            # Add zero padding to enable order-free layers.
            zeros = np.zeros(tuple(list(input_arr.shape[:3]) + [num_filters - 1]))
            input_arr = np.concatenate((input_arr, zeros), axis=3)

        # Create categorical output.
        num_classes = max(target_arr) + 1
        task['num_classes'] = num_classes
        print(num_classes)
        target_one_hot = to_categorical(target_arr, num_classes=num_classes)

        # Split into train and val
        input_train = []
        target_train = []
        input_val = []
        target_val = []
        counts = {}
        for i in range(input_arr.shape[0]):
            one_input = input_arr[i]
            one_output = target_arr[i]
            y_one_hot = target_one_hot[i]
            if one_output not in counts:
                counts[one_output] = 0
            if counts[one_output] < train_samples_per_character:
                input_train.append(one_input)
                target_train.append(y_one_hot)
                counts[one_output] += 1
            else:
                input_val.append(one_input)
                target_val.append(y_one_hot)

        task['X_train'] = np.array(input_train)
        task['Y_train'] = np.array(target_train)
        task['X_val'] = np.array(input_val)
        task['Y_val'] = np.array(target_val)

    return tasks
