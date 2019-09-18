# Text Classification

This is a generic domain for evolving text classifiers.
Text classifiers solve the problem of mapping a sequence of words, e.g., a sentence, paragraph, comment, or tweet, into one of two or more classes.

## Data Format

This domain uses two data files: `tokens.pkl` and `labels.pkl`, which are created following standard Keras practices (See https://keras.io/preprocessing/text/ or `domain/toxicity/preprocessing/preprocess_data_example.ipynb`, or `domain/toxicity/preprocessing/preprocess_data.py`).

`tokens.pkl` contains the input data.
This file is created by preprocessing (i.e., tokenizing) raw text data.
The file contains a dictionary of three data splits `train`, `dev`, and `test`.
Each of these splits contains a list of input samples.
Each input sample encodes a sequence of text (e.g., a sentence, paragraph, comment, or tweet) as a list of integers.
That is, input sample is a list of integers, and each integer corresponds to a word in the vocabulary.
The mapping from words to their corresponding integers is created during preprocessing, but is not needed by ENN, so it is not included as a data file.

As an example, suppose `tokens.pkl` is loaded to the variable `tokens`.
Then, if the integers for the words "the", "cat", and "jumped", are "1", "42", and "3016", respectively, and the raw text for the 10th training sample is "The cat jumped.", then the value of `tokens['train'][10]` would be `[1, 42, 3016]`.
I.e.,
```
import pickle
with open('tokens.pkl', 'rb') as f:
    tokens = pickle.load(f)
print(tokens['train'][10])

output: [1, 42, 3016]
```

`labels.pkl` contains the output data.
This file contains a dictionary of three data splits `train`, `dev`, and `test`.
Each of these splits consists of a list of integers.
Each integer corresponds to a class.
E.g., in the case of sentiment classification, there may be two classes: positive and negative.

## Domain-specific Configuration

See `"domain_config"` section of the hocon (e.g., `textclassifier/config/test_enn/test_config.hocon`).

### Dataset specification

The example dataset for this domain is wikipedia toxic comment classification, where each comment is to be classified into one of two classes: toxic or not toxic.
This dataset is specified in `"data_basedir"`.

To use a different dataset, upload the corresponding `tokens.pkl` and `labels.pkl` file to an accessible bucket, and change `"data_basedir"` to point to this location, and change `"num_classes"` to reflect the number of classes in the new dataset.

### Domain-specific parameters

`"max_sentence_length"` This is the maximum length of sequences fed into the model. Sequences in the dataset that are longer than this will be truncated to this length; sequences that are shorter will be zero-padded to this length.


