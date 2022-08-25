import matplotlib.pyplot as plt
import os
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.utils import text_dataset_from_directory

# supress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def plot(history_dict):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
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
    """
    Predict the label of stackoverflow question.
    Can be: csharp, java, javascript, python
    """

    batch_size = 32
    seed = 42
    raw_train_ds = text_dataset_from_directory(
        'stack_overflow_16k/train',
        batch_size=batch_size,
        subset='training',
        seed=seed,
        validation_split=0.2,
    )

    raw_val_ds = text_dataset_from_directory(
        'stack_overflow_16k/train',
        batch_size=batch_size,
        subset='validation',
        seed=seed,
        validation_split=0.2
    )

    raw_test_ds = text_dataset_from_directory(
        'stack_overflow_16k/test',
        batch_size=batch_size
    )

    max_features = 1000
    sequence_length = 250

    vectorize_layers = layers.TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layers(text), label

    train_text = raw_train_ds.map(lambda x, y: x)

    vectorize_layers.adapt(train_text)

    # See the contents of dataset tensor
    text_batch, label_batch = next(iter(raw_train_ds))
    text, label = text_batch[0], label_batch[0]
    print("review", text)
    print("label", label)
    print("vectorized review", vectorize_text(text, label))

    # Apply vectorization to dataset
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, 16),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(4),
    ])

    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

    history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[callback])

    loss, accuracy = model.evaluate(test_ds)

    print(f'Loss: {loss}\nAcc: {accuracy}')

    history_dict = history.history

    plot(history_dict)

    complete_model = tf.keras.Sequential([
        vectorize_layers,
        model,
        layers.Activation('softmax')
    ])
    complete_model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="adam", metrics=['accuracy']
    )
    complete_model.save('model', save_format='tf')

    labels = {0: 'csharp', 1: 'java', 2: 'javascript', 3: 'python'}
    with open('stack_overflow_16k/test/python/0.txt') as f:
        python_question = f.readlines()
    print(python_question)

    prediction = complete_model.predict([python_question]).argmax()

    print(labels[prediction])


if __name__ == '__main__':
    main()
