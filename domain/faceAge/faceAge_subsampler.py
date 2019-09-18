#!/usr/bin/env python
import random
import numpy as np

def softmax(value, tau):
    """Compute softmax values for each sets of scores in value."""
    exponent = np.exp(value / tau)
    return exponent / np.sum(exponent, axis=0)

def class_counts(partition, labels):
    """Count the number of diseases in the current partition"""
    num_classes = len(labels[partition[0]])
    total_positives = np.zeros(num_classes)
    total_negatives = 0
    for key in partition:
        labels_arr = np.array(labels[key])
        total_positives += labels_arr
        if 1 not in labels_arr:
            total_negatives += 1.
    print("Size of partition: {}".format(str(len(partition))))
    print("Total positives: {}".format(str(total_positives)))
    print("Total negatives: {}".format(str(total_negatives)))
    return total_positives, total_negatives, num_classes

# Public Enemy #13 for too-many-statements
# pylint: disable=too-many-statements
def stratified_subsample(partition, labels, subsample_amount, tau):
    """
    - partition is a list of keys, e.g., image names.
    - labels is a dictionary of keys to labels
        - each label is a list (or array) where there is a 1 in the ith location
          if the ith class is positive, 0 otherwise.
    - subsample_amount is the number of keys from the partition to return
    """

    # Get original count for each case.
    print("Original Counts")
    total_positives, total_negatives, num_classes = \
        class_counts(partition, labels)

    # Get target ratio for each class.
    positive_ratios = total_positives / len(partition)
    negative_ratio = np.array([total_negatives / len(partition)])
    ratios = np.concatenate((positive_ratios, negative_ratio))
    if tau is not None and tau > 0.0:
        ratios = softmax(ratios, tau)
    positive_ratios = ratios[:-1]
    negative_ratio = ratios[-1]

    print("")
    print("Positive ratios: {}".format(str(positive_ratios)))
    print("Negative ratio: {}".format(str(negative_ratio)))
    print("Total: {}".format(str(np.sum(positive_ratios) + negative_ratio)))

    # Get target count for each class.
    target_positives = (positive_ratios * subsample_amount).astype(int)
    target_negatives = int(negative_ratio * subsample_amount)

    print("")
    print("Target positives: {}".format(str(target_positives)))
    print("Target negatives: {}".format(str(target_negatives)))
    print("Total: {}".format(str(np.sum(target_positives) + target_negatives)))

    # Collect subsample by iterating over shuffled partition.
    shuffled_partition = list(partition)
    np.random.shuffle(shuffled_partition)
    partition_subsample = []
    sample_positives = np.zeros(num_classes)
    sample_negatives = 0
    for key in shuffled_partition:
        labels_arr = np.array(labels[key])
        keep = False
        for idx, value in enumerate(labels_arr):
            if value == 1.0:
                if sample_positives[idx] < target_positives[idx]:
                    keep = True
        if 1 not in labels_arr:
            if sample_negatives < target_negatives:
                keep = True
                sample_negatives += 1
        if keep:
            partition_subsample.append(key)
            sample_positives += labels_arr

    print("")
    print("Sample positives: {}".format(str(sample_positives)))
    print("Sample negatives: {}".format(str(sample_negatives)))
    print("Size of partition: {}".format(str(len(partition_subsample))))
    print("Total: {}".format(str(np.sum(sample_positives) + sample_negatives)))

    # Readjust size of subsampled parition to match subsampled amount
    if len(partition_subsample) > subsample_amount:
        partition_subsample = partition_subsample[:subsample_amount]
    else:
        while len(partition_subsample) < subsample_amount:
            new_entry = random.choice(partition)
            if new_entry not in partition_subsample:
                partition_subsample.append(new_entry)
    assert len(partition_subsample) == subsample_amount

    print("")
    print("Readjusted Counts")
    class_counts(partition_subsample, labels)

    # Return subample.
    return partition_subsample
