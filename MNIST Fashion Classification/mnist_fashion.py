import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from os.path import exists

# supress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# get tf version and see if gpu is enabled
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def plot_image(i, predictions_array, true_label, img, class_names):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'b'
    else:
        color = 'r'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_predictions(i, predictions, test_labels, test_images, class_names):
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images, class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()


def main():
    # train test split, 60k train images 10k test images
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # explore shape of data
    print(f'Data Shape: \n'
          f'Images{train_images.shape}\n'
          f'Labels: {train_labels.shape}')

    # preprocess data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # compile and run
    if exists('../MNIST Digit Classification/model.h5'):
        model = tf.keras.models.load_model('model.h5')
        print('Model Loaded Successfully!')
    else:
        # build the model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics='accuracy')
        model.fit(train_images, train_labels, epochs=10)
        model.save('model.h5')

    # evaluate model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(f'\nTest Accuracy: {test_acc}')

    # make predictions
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)

    # verify predictions
    plot_predictions(0, predictions, test_labels, test_images, class_names)


if __name__ == '__main__':
    main()
