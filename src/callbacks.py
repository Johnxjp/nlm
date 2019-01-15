from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def checkpoint(file, period=1):
    return ModelCheckpoint(
        file,
        save_weights_only=True,
        save_best_only=True,
        period=period,
    )


def early_stopping(patience=5):
    return EarlyStopping(
        patience=patience,
    )