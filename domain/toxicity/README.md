# Toxic Comment Classification

This domain is a specialization of the `textclassifier` domain.
It is also fairly generic, and includes some additional functionality that expands the ENN design space.
There are two main differences between this domain and `textclassifier`:
1. ENN has the option to use pre-trained word embeddings if they are found to be useful.
2. ENN has access to a larger set of global training hyperparameters to evolve.


## Data Format

See `domain/toxicity/preprocessing/preprocess_data_example.ipynb` for an example of how this data is preprocessed.

The input and output data format are equivalent to that used by textclassifier (see `domain/textclassifier/README.md`).
The output is specified in `labels.pkl`, and the input is specified in `toxicity_tokens_<vocab_size>_words.pkl`, where `<vocab_size>` is defined below in Domain-specific parameters.
As an example, suppose `toxicity_tokens_150000_words.pkl` is loaded to the variable `tokens`.
Then, if the integers for the words "the", "cat", and "jumped", are "1", "42", and "3016", respectively, and the raw text for the 10th training sample is "The cat jumped.", then the value of `tokens['train'][10]` would be `[1, 42, 3016]`.
I.e.,
```
import pickle
with open('tokens.pkl', 'rb') as f:
    tokens = pickle.load(f)
print(tokens['train'][10])

output: [1, 42, 3016]
```

In addition to input and output data, this domain makes use of pre-trained word embeddings.
ENN can choose to use these during evolution if they are effective.
The files containing these embeddings are stored in the same bucket as `tokens.pkl` and `labels.pkl`.
Each embeddings file is of the form `<kind>_matrix_<vocab_size>.pkl`, where `<kind>` is either `fasttext` or `glove`, and `<vocab_size>` is the number of words represented by the embedding (e.g., `150000`).
The embedding for each word is a length 300 vector of floats, so loading the embedding file results in a numpy array of shape (<vocab_size>,300).
Each time a model is trained, the embedding will be trimmed down to the evolved vocab size.
For an example of how to create such embedding files during preprocessing, see `domain/toxicity/preprocessing/preprocess_data_example.ipynb` or `domain/toxicity/preprocessing/preprocess_data.py`.
For reference on how embeddings are incorporated into the model, see `apply_embedding_encoder` in `domain/toxicity/toxicity_evaluator.py`.


## Domain-specific Configuration


### domain_config

See `"domain_config"` section of the hocon (e.g., `toxicity/config/record_breaking_config.hocon`).

`"max_sentence_length"` This is the maximum length of sequences fed into the model. Sequences in the dataset that are longer than this will be truncated to this length; sequences that are shorter will be zero-padded to this length.

`"vocab_size"` This is the maximum number of unique words included in the preprocessed input data. This vocab size used during training can be evolved to be any number less than this.

`"evolve_embeddings"` When set to `true`, this allows evolution to decide which pre-trained embeddings to use, if any.


### builder_config

For added flexibility, this domain allows additional training hyperparameters to be evolved.
See `"evaluation_hyperparameter_spec"` in `"builder_config"` section of the hocon (e.g., `toxicity/config/record_breaking_config.hocon`).

`"embeddings"` Options for what kind of embedding to use.

`"embeddings_trainable"` Whether pre-trained embeddings should be kept fixed, or be allowed to be fine-tuned during training.

`"vocab_size"` How many unique words to include in the vocab (preferenced by frequency). Larger vocab sizes have higher expressivity at the possible cost of worse generalization for uncommon words.

`"epoch_training_percentage"` Allows evolution some flexibility in how many training steps it takes. These models can overfit quickly, so they can be sensitive to such changes in training amount.

`"input_noise"` Option to add Gaussian noise to input to reduce overfitting and potentially improve generalization.


