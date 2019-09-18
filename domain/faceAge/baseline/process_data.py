#!/usr/bin/env python

import glob
import os

from itertools import chain

import h5py
from tqdm import tqdm
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras_preprocessing.image import load_img, img_to_array
from skimage.util import montage
import matplotlib.pyplot as plt

# Max number of samples in dataset
MAX_SAMPLES = 112120
# Images base directory
IMG_BASEDIR = "/home/jason/.kaggle/datasets/nih-chest-xrays/data/images"
# Path to csv file for x-ray
XRAY_CSV_FILE = "/home/jason/.kaggle/datasets/nih-chest-xrays/data/Data_Entry_2017.csv"
# Number of samples to get (max for x-ray dataset is 112,120)
NUM_SAMPLES = 1000
NUM_SAMPLES = min(MAX_SAMPLES, NUM_SAMPLES)
# Dimension for images when given as input to model
IMG_SIZE = 224
# Directory to output hdf5 to
OUTDIR = 'output'
# Name of h5 file data for images and labels
H5_FILE_NAME = 'chest_xray_%s_%s.h5' % (NUM_SAMPLES, IMG_SIZE)
# Compression format for hdf5 file, None for no compression, gzip otherwise
COMPRESSION = 'gzip'


def write_df_as_hdf(out_path, out_df, compression='gzip'):
    with h5py.File(out_path, 'w') as h5_file:
        for k, arr_dict in tqdm(list(out_df.to_dict().items())):
            try:
                s_data = np.stack(list(arr_dict.values()), 0)
                try:
                    h5_file.create_dataset(k, data=s_data, compression=
                    compression)
                except TypeError as exception:
                    try:
                        h5_file.create_dataset(k, data=s_data.astype(np.string_),
                                         compression=compression)
                    except TypeError as exception2:
                        print("{0} could not be added to hdf5, {1} {2}".format(
                            k, repr(exception), repr(exception2)))
            except ValueError as exception:
                print("{0} could not be created, {1}".format(k, repr(exception)))
                all_shape = [np.shape(x) for x in list(arr_dict.values())]
                print('Input shapes: {}'.format(all_shape))


def print_h5(h5_file):
    # show what is inside
    with h5py.File(h5_file, 'r') as h5_data:
        for c_key in list(h5_data.keys()):
            print("{0} {1} {2}".format(c_key, h5_data[c_key].shape, h5_data[c_key].dtype))


def vis_h5(h5_file, num=36):
    # visualize h5 file as sanity check
    with h5py.File(h5_file, 'r') as h5_data:
        images = np.array(h5_data.get('images'))
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    ax1.imshow(montage(images[:num, :, :, 0])) #, cmap='bone')
    fig.savefig(os.path.join(OUTDIR, 'images.png'), dpi=300)


def imread_and_normalize(im_path):
    img = load_img(im_path, target_size=(IMG_SIZE, IMG_SIZE))
    n_img = img_to_array(img)
    n_img = n_img.astype(np.uint8)
    return np.expand_dims(n_img, -1)


def load_data():
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    h5_file_path = os.path.join(OUTDIR, H5_FILE_NAME)

    all_xray_df = pd.read_csv(XRAY_CSV_FILE)
    all_image_paths = {os.path.basename(x): x for x in
                       glob.glob(os.path.join(IMG_BASEDIR, '*.png'))}

    print("Scans found: {0}, Total Headers {1}".format(
                len(all_image_paths), all_xray_df.shape[0]))
    all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
    # all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x:
    #   int(x[:-1]))
    print(all_xray_df.sample(3))

    all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(
        lambda x: x.replace('No Finding', ''))
    all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(
        lambda x: x.split('|')).tolist())))
    print("All Labels {}".format(all_labels))
    for c_label in all_labels:
        if len(c_label) > 1:  # leave out empty labels
            all_xray_df[c_label] = all_xray_df['Finding Labels'].map(
                lambda finding, l_label=c_label: 1.0 if l_label in finding else 0)
    print(all_xray_df.sample(3))

    # since we can't have everything make a nice subset
    # weight is 0.1 + number of findings
    # sample_weights = all_xray_df['Finding Labels'].map(lambda x:
    #   len(x.split('|')) if len(x)>0 else 0).values + 1e-1
    # sample_weights /= sample_weights.sum()
    all_xray_df = all_xray_df.sample(min(all_xray_df.shape[0], NUM_SAMPLES))
    write_df_as_hdf(h5_file_path, all_xray_df, compression=COMPRESSION)
    # print_h5(h5_file_path)

    # test_img = imread_and_normalize(all_xray_df['path'].values[0])
    # print test_img.shape
    out_image_arr = np.zeros((all_xray_df.shape[0],) + (IMG_SIZE, IMG_SIZE, 1),
                             dtype=np.uint8)
    for i, c_path in enumerate(tqdm(all_xray_df['path'].values)):
        out_image_arr[i] = imread_and_normalize(c_path)
    with h5py.File(h5_file_path, 'a') as h5_data:
        h5_data.create_dataset('images', data=out_image_arr,
                               compression=COMPRESSION)
    print_h5(h5_file_path)
    # vis_h5(h5_file_path)

    print("Output File-size {0:2.2f}MB".format(os.path.getsize(h5_file_path) / 1e6))


if __name__ == "__main__":
    load_data()
    # print_h5('output/chest_xray_128.h5')
