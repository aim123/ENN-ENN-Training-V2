#
# Functions for adding decoders to models.
#

from keras.layers import Concatenate, Dense
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D

# This is a currently popular (circa 6/18) decoder for NLP.
# Note: It also happens to be non-sequential.
def apply_concat_pooling_1d_decoder(input_tensor, num_classes):
    avg_pool = GlobalAveragePooling1D()(input_tensor)
    max_pool = GlobalMaxPooling1D()(input_tensor)
    conc_pool = Concatenate()([avg_pool, max_pool])
    output_tensor = Dense(num_classes, activation="sigmoid")(conc_pool)
    return output_tensor
