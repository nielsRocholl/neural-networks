from enum import auto
import os
import warnings
import pickle as pkl
import numpy as np
import umap as u_map

from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.engine.input_layer import InputLayer
from keras.layers import Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from helper_functions.preprocessing import preprocess
from helper_functions.img_scatterplot import scatterplot_with_imgs
from helper_functions.helpers import plot, pad_mcmc, neighborhood_hit
from helper_functions.from_raw_to_pickle_mcmc import get_pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle

# suppress warnings
from helper_functions.unlabeled_data_pickle_flow import extract_unlabelled

warnings.filterwarnings('ignore')


def task_1():
    print('Loading data...')
    data = {}
    if os.path.exists(f'pkl{os.sep}data_mcmc.pkl'):
        print('Loading data from pickle file\n')
        data = pkl.load(open(f'pkl{os.sep}data_mcmc.pkl', 'rb'))
    else:
        # Adding labelled data to the dataset
        raw_data = pkl.load(open(f'pkl{os.sep}mcmc_labelled_data.pkl', 'rb'))
        names = pkl.load(open(f'pkl{os.sep}mcmc_labelled_names.pkl', 'rb'))

        # Reshaping such that it fits better with code that already exists for task 1-3
        raw_data = raw_data.reshape(raw_data.shape[0], raw_data.shape[1], raw_data.shape[2], 1)

        # Shuffeling the data otherwise the subset selection that is made later does not select datapoints from all labels.
        # Shuffeling with the same key, otherwise the labels are lost.
        raw_data, names = shuffle(raw_data, names)

        # Adding to the dictionary with all the data
        data['labelled_names'] = names
        data['labelled_data'] = dict(zip(['normalized_data', 'data_mean', 'data_std'],
                                         preprocess(data=raw_data, visualize=True, cropping=False, mcmc=True)))

        # Adding unlabelled data to the dataset
        raw_data = pkl.load(open(f'pkl{os.sep}mcmc.pkl', 'rb'))

        # Also shuffeling unlabelled data
        raw_data = shuffle(raw_data)

        # Reshaping such that it fits better with code that already exists for task 1-3
        raw_data = raw_data.reshape(raw_data.shape[0], raw_data.shape[1], raw_data.shape[2], 1)

        data['unlabelled_data'] = dict(zip(['normalized_data', 'data_mean', 'data_std'],
                                           preprocess(data=raw_data, visualize=True, cropping=False, mcmc=True)))

        pkl.dump(data, open(f'pkl{os.sep}data_mcmc.pkl', 'wb'))
    return data


def plot_data(data, data_reduced: np.array, labels: list, method: str, neigh_hit: float) -> None:
    """
    Plot both the original and reduced data
    :param data: original data
    :param data_reduced: reduced data
    :param labels: labels of data
    :param method: name of method used for dimensionality reduction
    :param neigh_hit: neighborhood hit score
    """

    plt.clf()
    fig, ax = plt.subplots(ncols=2, figsize=(25, 7.5), dpi=300)
    # plot original data
    ax[0] = scatterplot_with_imgs(data_reduced[:, 0], data_reduced[:, 1], data, ax=ax[0], zoom=0.1)
    # create list for label counter
    label_counter = [[], [], [], [], []]
    for idx, item in enumerate(data_reduced):
        count = 1
        while 6 > count:
            if labels[idx] == [str(count)]:
                label_counter[count - 1].append(item)
            count += 1

    # plot all different data label counters
    for idx, counter in enumerate(label_counter):
        if len(counter) > 0:
            ax[1].scatter(np.array(counter)[:, 0], np.array(counter)[:, 1], label=str(idx + 1))

    # add title and legend
    ax[0].set_title(f'Original data', fontsize=25)
    ax[1].set_title(f'{method} data\nneigh hit: {neigh_hit}', fontsize=25)
    plt.legend(prop={'size': 20})
    plt.savefig(f'plots{os.sep}{method}MCMC.png')
    plt.show()


def task_2(preprocessed_data):
    """
    Load data from pickle file and perform dimensionality reduction:
    - PCA
    - t-SNE
    - UMAP
    :param preprocessed_data: dictionary containing preprocessed data
    """

    def pca(data: np.array) -> np.array:
        d = data[:, :, :, 0]
        d = d.reshape(d.shape[0], d.shape[1] * d.shape[2])
        method = PCA(n_components=2)
        return method.fit_transform(d)

    def tsne(data: np.array) -> np.array:
        d = data[:, :, :, 0]
        d = d.reshape(d.shape[0], d.shape[1] * d.shape[2])
        method = TSNE(n_components=2)
        return method.fit_transform(d)

    def umap(data: np.array) -> np.array:
        d = data[:, :, :, 0]
        d = d.reshape(d.shape[0], d.shape[1] * d.shape[2])
        method = u_map.UMAP(n_components=2)
        return method.fit_transform(d)

    data = preprocessed_data['labelled_data']['normalized_data']
    labels = preprocessed_data['labelled_names']
    labels = [label.split(' ') for label in labels]

    # Perform all dimensionality reduction methods
    print('Performing Dimensionality Reduction Methods...')
    for algo in [pca, tsne, umap]:
        print(f'Performing {algo.__name__}')
        data_reduced = algo(data)
        neigh_hit = neighborhood_hit(data_reduced, labels).__round__(2)
        print(f'Neighborhood hit: {neigh_hit}')
        plot_data(data, data_reduced, labels, algo.__name__, neigh_hit=neigh_hit)


class AutoEncoder(Model):
    """
    Autoencoder class
    Encoder and decoder are symmetric
    """

    def __init__(self, input_shape, latent_dim):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Sequential([
            InputLayer(input_shape=input_shape),
            Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
            Flatten(),
            Dense(latent_dim, kernel_regularizer='l1_l2')
        ])
        self.decoder = Sequential([
            InputLayer(input_shape=(latent_dim,)),
            Dense(units=16 * 16 * latent_dim),
            Reshape(target_shape=(16, 16, latent_dim)),
            Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='linear'),
            Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='linear'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def task_3(preprocessed_data):
    # Selecting 5000 training samples
    # And 1000 testing samples, as described in the assignment.
    train = preprocessed_data['unlabelled_data']['normalized_data'][:5000]
    test = preprocessed_data['labelled_data']['normalized_data'][:1000]
    labels = preprocessed_data['labelled_names'][:1000]
    labels = [label.split(' ') for label in labels]
    # Train shape is (5000, 50, 50, 1). Test shape is (1000, 50, 50, 1)
    # The shapes need to be divisable twice, so they are padded to (p, 64, 64, 1)

    train = pad_mcmc(train)
    test = pad_mcmc(test)

    print(f'Train shape: {train.shape}')

    autoencoder = AutoEncoder(input_shape=train.shape[1:], latent_dim=16)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    autoencoder.fit(train, train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    latent_space_data = autoencoder.encoder.predict(test)
    decoder = autoencoder.decoder.predict(latent_space_data)

    print('Plotting original data...')
    plot(data=test[:50], title='Original Data MCMC')
    print('Plotting reconstructed data...')
    plot(decoder, title='Reconstructed Data MCMC')
    print('Plotting latent space data...')
    neigh_hit = neighborhood_hit(latent_space_data, labels).__round__(2)
    print(f'Neighborhood hit: {neigh_hit}')
    plot_data(test, latent_space_data, labels, 'Latent Space Data', neigh_hit=neigh_hit)


def main():
    if not os.path.exists('pkl'):
        os.mkdir('pkl')
    if not os.path.exists(f'pkl{os.sep}mcmc.pkl'):
        print('Unlabelled pickle data file for task 4 does not exist, extracting data')
        get_pickle(labelled=False)
    if not os.path.exists(f'pkl{os.sep}mcmc_labelled_data.pkl'):
        print('labelled pickle data file for task 4 does not exist, extracting data')
        get_pickle(labelled=True)

    # Task 4 essentialy entails doing tasks 1-3 over, but with a new dataset.
    # Hence the functions have the same setups/names
    # They do differ in small aspects, hence being copied here.
    preprocessed_data = task_1()
    task_2(preprocessed_data)
    task_3(preprocessed_data)


if __name__ == '__main__':
    main()
