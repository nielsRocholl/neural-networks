from os.path import exists

import matplotlib.pyplot as plt
import os
import re
import shutil
import string

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

# supress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


def plot(history_dict):
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()


def main():
    # extract data
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')

    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

    print(f'Dataset dir: {os.listdir(dataset_dir)}')

    train_dir = os.path.join(dataset_dir, 'train')

    print(f'Train dir: {os.listdir(train_dir)}')

    # remove unnecessary folder
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

    # create train, val, test split
    batch_size = 32
    seed = 42

    # load data
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )

    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test',
        batch_size=batch_size)

    # create text vectorization
    max_features = 10000
    sequence_length = 250

    print(raw_train_ds)


    vectorize_layers = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layers(text), label

    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layers.adapt(train_text)

    text_batch, label_batch = next(iter(raw_train_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print("review", first_review)
    print("label", first_label)
    print("vectorized review", vectorize_text(first_review, first_label))

    # apply vectorization to train, test and validation data
    train_ds = raw_train_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # create the model
    embedding_dim = 16

    # load if exists
    if exists('model.h5'):
        model = tf.keras.models.load_model('model.h5')
        history = np.load('history.npy', allow_pickle=True).item()
        print('Model Loaded Successfully!')
    # compile and run
    else:
        model = tf.keras.Sequential([
            layers.Embedding(max_features + 1, embedding_dim),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])
        print(model.summary())
        model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                      optimizer='adam',
                      metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
        history = model.fit(train_ds, validation_data=val_ds, epochs=10)
        np.save('history.npy', history)
        model.save('model.h5')

    loss, accuracy = model.evaluate(test_ds)

    print(f'Loss: {loss}\nAccuracy: {accuracy}')

    history_dict = history.history
    print(history_dict.keys())

    plot(history_dict)

    # create model that incorporates vectorization of input
    compile_model = tf.keras.Sequential([
        vectorize_layers,
        model,
        layers.Activation('sigmoid')
    ])

    compile_model.compile('model', save_format='tf')





if __name__ == '__main__':
    main()
