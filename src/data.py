"""Data functions"""

import numpy as np
from sklearn.model_selection import train_test_split

from src import preprocessing
from src.utils import utils_io


def subsequences(sequence, length):
    """
    Returns sequence of length n, plus next word token for label

    :param sequence: tokenised string
    :param length: length of subsequence
    """

    if length >= len(sequence):
        return None

    i = 0
    sequences, labels = [], []
    while i + length + 1 < len(sequence):
        sequences.append(sequence[i: i + length])
        labels.append(sequence[i + length])
        i += 1

    return sequences, labels


def build(file, maxlen, word2id, test_split=0.2, val_split=0.1, shuffle=True, seed=50):
    """Returns x, y as a sequence of maxlen and a single word to predict"""

    temp = utils_io.load_clean(file)

    data = []
    for t in temp:
        data.append(f"<start> {t} <eos>")

    data = preprocessing.tokenise(" ".join(data))
    x, y = subsequences(data, maxlen)
    x, y = convert_to_int(x, word2id), convert_to_int(y, word2id)

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=test_split, shuffle=shuffle, random_state=seed)

    if val_split is not None or val_split != 0:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                test_size=val_split, random_state=seed)
        return (np.asarray(x_train), np.asarray(y_train)), \
               (np.asarray(x_val), np.asarray(y_val)), \
               (np.asarray(x_test), np.asarray(y_test))

    return (np.asarray(x_train), np.asarray(y_train)), \
           (np.asarray(x_test), np.asarray(y_test))


def convert_to_int(sequences, word2id, unk_token="<unk>"):

    unk = word2id[unk_token]
    if isinstance(sequences[0], list):
        return [[word2id[w] if word2id.get(w, False) else unk for w in sent] for sent in sequences]

    return [word2id[w] if word2id.get(w, False) else unk for w in sequences]
