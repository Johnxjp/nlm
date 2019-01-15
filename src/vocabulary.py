"""Builds vocabulary"""

from operator import itemgetter

import numpy as np


def count_frequency(tokenised_data):
    """Array of tokenised sentences"""

    vocab = {}
    for sent in tokenised_data:
        for word in sent:
            vocab[word] = vocab.get(word, 0) + 1

    return vocab


def sort_by_val(dict_input, reverse=True):
    return sorted(dict_input, key=itemgetter(1), reverse=reverse)


def load_vocab(embedding_file, vocab_size=None):

    word2id = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<eos>": 3}
    n_special_tokens = len(word2id)

    with open(embedding_file, 'r', encoding='utf-8', errors='ignore') as f:

        # First row contains meta information
        n, d = map(int, f.readline().strip().split())

        if vocab_size is not None:
            embedding_matrix = np.zeros((vocab_size, d))
        else:
            embedding_matrix = np.zeros((n, d))

        for row, line in enumerate(f):
            if vocab_size is None or n_special_tokens + row < vocab_size:
                tokens = line.rstrip().split()
                word2id[tokens[0]] = n_special_tokens + row
                embedding_matrix[n_special_tokens + row] = np.fromiter(
                    map(float, tokens[1:]), dtype=np.float16)

    # Assign vectors for special values
    for i in range(n_special_tokens):
        embedding_matrix[i][i] = 1

    return embedding_matrix, word2id