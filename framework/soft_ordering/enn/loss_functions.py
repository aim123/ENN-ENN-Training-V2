import keras.backend as K

# Custom loss function definitions

def unweighted_binary_crossentropy(y_true, y_pred):
    """
    Custom loss function for multi-label problems.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: the sum of binary cross entropy loss across all the classes
    """
    return K.sum(K.binary_crossentropy(y_true, y_pred))
