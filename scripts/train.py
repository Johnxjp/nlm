"""Training script"""
import os

import numpy as np

from src import data, model, callbacks
from src.utils import utils_io


def main():

    configs = utils_io.json_load(utils_io.get_full_path("configs.json", "src"))

    # Build Model
    m = model.build_model(
        configs['data_config']['maxlen'],
        configs['data_config']['vocab_size'],
        utils_io.pickle_load(utils_io.get_full_path(
            "embeddings/embedding_matrix_20000.p", "root")),
        configs['nlm']['hidden_dims'],
        configs['nlm']['dropout']
    )

    print(m.summary())

    # Load Data
    training, validation, testing = data.build(
        utils_io.get_full_path("data/interim/book_1.txt", "root"),
        configs['data_config']['maxlen'],
        utils_io.pickle_load(utils_io.get_full_path(
            "embeddings/word2id_20000.p", "root")),
        seed=configs['data_config']['split_seed']
    )

    print(list(map(np.shape, (training[0], training[1], validation[0], testing[0]))))

    # Compile
    train_configs = configs['train_config']
    m.compile(
        optimizer=train_configs['optimizer'],
        loss=model.custom_loss,
        metrics=['sparse_categorical_accuracy'],
    )

    # Define Save Path
    i = 0
    save_path = utils_io.get_full_path(train_configs['output_directory'], "root")
    while os.path.exists(f"{save_path}/model_{i}.h5"):
        i += 1

    save_name = f"model_{i}"

    # Train
    history = m.fit(
        x=training[0],
        y=training[1],
        batch_size=train_configs['batch_size'],
        epochs=train_configs['epochs'],
        validation_data=validation,
        callbacks=[
            callbacks.checkpoint(
                utils_io.get_full_path(
                    f"{train_configs['checkpoint_path']}/{save_name}.h5",'root')),
            callbacks.early_stopping(patience=5)
        ]
    )

    # Save
    utils_io.pickle_save(history, f"{save_path}/{save_name}-history.p")
    utils_io.json_save(configs, f"{save_path}/{save_name}-configs.json")


if __name__ == '__main__':
    main()