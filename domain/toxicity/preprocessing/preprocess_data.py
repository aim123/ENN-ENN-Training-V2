'''
    Script for word-based tokenization of toxicity data.
'''

import pickle

import sys

from keras_preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

# Read commandline args.

class PreprocessData():

    def run(self):

        # Path to toxicity_annotated_comments.merged.shuf.cleaned-68MB,_160k-rows.tsv
        raw_data_file = sys.argv[1]

        vocab_size = int(sys.argv[2])

        # Optional path to embeddings file to create embeddings matrix.
        if len(sys.argv) > 3:
            embeddings_file = sys.argv[3]
        else:
            embeddings_file = None

        # Load raw data.
        print("Loading raw data")
        raw_df = pd.read_csv(raw_data_file, sep='\t')

        # Split data.
        print("Splitting data")
        splits = ['train', 'test', 'dev']
        split_labels = {}
        raw_split_text = {}
        for split in splits:
            labels = raw_df.loc[raw_df['split'] == split]['Label'].tolist()
            text = raw_df.loc[raw_df['split'] == split]['comment'].tolist()
            split_labels[split] = labels
            raw_split_text[split] = text

        # Setup tokenizer.
        print("Setting up tokenizer.")
        all_text = raw_df['comment'].tolist()
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(all_text)

        # Tokenize text.
        print("Tokenizing")
        split_text = {}
        for split in splits:
            print(("Tokenizing", split))
            tokenized = tokenizer.texts_to_sequences(raw_split_text[split])
            split_text[split] = tokenized

        # Save to tokens file.
        print("Saving to file")
        with open('toxicity_labels.pkl', 'w') as my_file:
            pickle.dump(split_labels, my_file)
        with open('toxicity_tokens_{}_words.pkl'.format(vocab_size), 'w') as my_file:
            pickle.dump(split_text, my_file)

        # Create embeddings matrix
        # Code copied from https://www.kaggle.com/tunguz/bi-gru-lstm-cnn-poolings-fasttext.
        if embeddings_file is not None:
            max_features = vocab_size
            embed_size = 300
            print("Loading embeddings")

            embedding_index = dict(self.get_coefs(*o.strip().split(" "))
                           for o in open(embeddings_file))

            print("Creating embedding matrix")
            word_index = tokenizer.word_index
            nb_words = min(max_features, len(word_index))
            embedding_matrix = np.zeros((nb_words, embed_size))
            for word, i in list(word_index.items()):
                if i >= max_features:
                    continue
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            embeddings_matrix_filename = 'embeddings_matrix_{}.pkl'.format(vocab_size)
            with open(embeddings_matrix_filename, 'w') as my_file:
                pickle.dump(embedding_matrix, my_file)

        print("Done.")


    def get_coefs(self, word, *arr):
        return word, np.asarray(arr, dtype='float32')


if __name__ == "__main__":
    APP = PreprocessData()
    APP.run()
