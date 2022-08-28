import pathlib
import numpy as np

from preprocessing.image_preprocessing_helper_functions import *


def image_input_pipeline(data_dir=None):
    """
    Custom input pipeline using tf.data. Beginning from a TGZ file.
    Only Works if files are 2 dirs deep from 'data_dir
    """
    assert data_dir is not None
    # check if files are 2 dirs deep
    assert check_file_location(data_dir)

    data_dir = pathlib.Path(data_dir)

    # Get all image paths
    list_ds = tf.data.Dataset.list_files(f'{data_dir}{os.sep}*{os.sep}*', shuffle=False)
    image_count = len(list_ds)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))

    # Split dataset into training and validation
    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    # Create a dataset of image, label pairs
    # Set 'num_parallel_calls' so multiple images are loaded/processed in parallel.
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x: process_path(x, class_names), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x: process_path(x, class_names), num_parallel_calls=AUTOTUNE)

    # normalize rgb channel
    # train_ds = normalize(train_ds)
    # val_ds = normalize(val_ds)

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    return train_ds, val_ds
