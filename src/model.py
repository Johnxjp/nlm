"""Builds a model"""

from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.backend import sparse_categorical_crossentropy


def build_model(input_dims, vocab_size, embedding_matrix, hidden_dims, dropout):

    inps = Input((input_dims,), name="inputs")

    x = Embedding(vocab_size,
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix],
                  trainable=False,
                  input_length=input_dims,
                  mask_zero=True)(inps)

    x = LSTM(hidden_dims)(x)

    x = Dropout(rate=dropout)(x)

    outputs = Dense(vocab_size, name="outputs")(x)

    return Model(inps, outputs)


def custom_loss(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)