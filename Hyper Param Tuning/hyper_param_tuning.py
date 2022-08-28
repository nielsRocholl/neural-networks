import os
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras

# supress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# get tf version and see if gpu is enabled
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # tune the number of units in the first dense layer
    # Choose an optimal value between 32 and 512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def main():
    # download data
    (img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

    # normalize pixel values
    img_train = img_train.astype('float32') / 255.0
    img_test = img_test.astype('float32') / 255.0

    # initiate tuner and perform hyper-tuning
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='hp',
                         project_name='hp_tuning')

    # create a callback to stop training early after reaching certain val_loss
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    # build model with the optimal hyperparamters
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # re-initialize and train again with optimal number of epochs
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)

    # evaluate hypermodel on test data
    eval_results = hypermodel.evaluate(img_test, label_test)
    print('[test loss, test accuracy]: ', eval_results)


if __name__ == '__main__':
    main()
