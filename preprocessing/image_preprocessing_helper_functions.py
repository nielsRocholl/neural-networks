import os
import tensorflow as tf


def check_file_location(data_dir):
    """
    Check if the files are 2 directories deep from 'data_dir'
    """
    first_subdir = os.listdir(f'{data_dir}{os.sep}')[0]
    firs_file = os.listdir(f'{data_dir}{os.sep}{first_subdir}')[0]
    path = f'{data_dir}{os.sep}{first_subdir}{os.sep}{firs_file}'
    return os.path.isfile(path)


def get_label(file_path, class_names):
    """
    Returns label of data sample as integer.
    Integer represents index of the label.
    """
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)


def decode_img(img):
    """
    Returns the image in the form of a resized 3D tensor.
    Only works on .jpeg
    """
    img_height = 180
    img_width = 180
    # Convert the compressed string to a 3D unit8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path, class_names):
    """
    Calls get_label and decode_img to convert a file path into
    a processed image label pair.
    """
    label = get_label(file_path, class_names)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def configure_for_performance(ds):
    """
    To train a model you want the data to be:
    - Well shuffled
    - Batched
    - Batches to be available ASAP
    We use tf data to add these features
    """
    AUTOTUNE = tf.data.AUTOTUNE
    batch_size = 32
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
