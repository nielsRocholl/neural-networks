import os
import warnings
import pickle as pkl
import numpy as np
import umap as u_map

from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.engine.input_layer import InputLayer
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose, ZeroPadding2D, Lambda, Dropout
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from helper_functions.img_scatterplot import scatterplot_with_imgs
from helper_functions.labeled_data_pickle_flow import extract_labelled
from helper_functions.preprocessing import preprocess
from helper_functions.helpers import plot, pad_and_scale, neighborhood_hit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# suppress warnings
from helper_functions.unlabeled_data_pickle_flow import extract_unlabelled

warnings.filterwarnings('ignore')


def task_1() -> dict:
    """
    Load unlabelled data from pickle file and place it in a dictionary in the form of:
    data = {
             'labelled_names': [],
             'unlabelled_names': [],
             'labelled_data': {'normalized_cropped_data': [], 'data_mean': x, 'data_std': x},
             'unlabelled_data: {'normalized_cropped_data': [], 'data_mean': x, 'data_std': x}
          }
    Perform preprocessing during loading
    Save to pickle file (then we dont have to perform preprocessing again)
    """
    print('Loading data...')
    if os.path.exists(f'pkl{os.sep}data.pkl'):
        print('Loading data from pickle file\n'
              'No Visualization')
        data = pkl.load(open(f'pkl{os.sep}data.pkl', 'rb'))
    else:
        data = {}
        for file in tqdm(os.listdir(f'pkl{os.sep}')):
            if file == 'mcmc.pkl' or file == 'data_mcmc.pkl' or file == 'mcmc_labelled_data.pkl' or file == 'mcmc_labelled_names.pkl':
                continue
            # Load raw data
            raw_data = pkl.load(open(f'pkl{os.sep}' + file, 'rb'))
            # Get name of folder
            name = file.replace('sampled-300_', '').replace('.pkl', '')
            if 'data' in name:
                # Preprocess data (crop and normalize) and add data to dictionary
                data[name] = dict(zip(
                    ['normalized_cropped_data', 'data_mean', 'data_std'],
                    preprocess(data=raw_data, visualize=True, cropping=True, mcmc=False)
                ))
            else:
                # Add names to dictionary
                data[name] = raw_data
        # Save to pickle file
        pkl.dump(data, open(f'pkl{os.sep}data.pkl', 'wb'))
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
    # create list for labels
    laminar = []
    turbulent = []
    for idx, item in enumerate(data_reduced):
        if labels[idx] == 'l':
            laminar.append(item)
        elif labels[idx] == 't':
            turbulent.append(item)
        else:
            print('No label!')
    # plot laminar and turbulent data
    ax[1].scatter(np.array(laminar)[:, 0], np.array(laminar)[:, 1], label='laminar')
    ax[1].scatter(np.array(turbulent)[:, 0], np.array(turbulent)[:, 1], label='turbulent')
    # add title and legend
    ax[0].set_title(f'Original data', fontsize=25)
    ax[1].set_title(f'{method} data\nneigh hit: {neigh_hit}', fontsize=25)
    plt.legend(prop={'size': 20})
    plt.savefig(f'plots{os.sep}{method}.png')
    plt.show()


def task_2(preprocessed_data: dict) -> None:
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

    data = preprocessed_data['labelled_data']['normalized_cropped_data']
    labels = preprocessed_data['labelled_names']
    labels = [label.split(' ')[1] for label in labels]

    # perform all dimensionality reduction methods
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
            Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
            Flatten(),
            Dense(latent_dim, kernel_regularizer='l1_l2')
        ])
        self.decoder = Sequential([
            InputLayer(input_shape=(latent_dim,)),
            Dense(units=20 * 6 * latent_dim),
            Reshape(target_shape=(20, 6, latent_dim)),
            Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='linear'),
            Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='linear'),
            Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='linear'),
            Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='linear'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def task_3(preprocessed_data: dict) -> None:
    """
    Create autoencoder for dimensionality reduction.
    train on unlabelled data, test on labelled data, plot results
    :param preprocessed_data:
    """
    # load train and test data
    train = preprocessed_data['unlabelled_data']['normalized_cropped_data']
    test = preprocessed_data['labelled_data']['normalized_cropped_data']
    # change shape from (300, 309, 84, 1) to (300, 320, 96, 1) and scale (scaling done in task 1 seems insufficient)
    # TODO: This is a bit hacky, but shapes need to be devisable by 2
    train = pad_and_scale(train)  # this is my own function, see helpers.py
    test = pad_and_scale(test)
    print(f'Train shape: {train.shape}')
    # create autoencoder for dimensionality reduction
    autoencoder = AutoEncoder(input_shape=train.shape[1:], latent_dim=64)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    # train autoencoder with EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    autoencoder.fit(train, train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    # test autoencoder
    latent_space_data = autoencoder.encoder.predict(test)
    decoder = autoencoder.decoder.predict(latent_space_data)
    print('Plotting original data...')
    plot(data=test[:50], title='Original Data')
    print('Plotting reconstructed data...')
    plot(decoder, title='Reconstructed Data')
    print('Plotting latent space data...')
    labels = preprocessed_data['labelled_names']
    labels = [label.split(' ')[1] for label in labels]
    neigh_hit = neighborhood_hit(latent_space_data, labels).__round__(2)
    print(f'Neighborhood hit: {neigh_hit}')
    plot_data(test, latent_space_data, labels, 'Latent Space Data', neigh_hit=neigh_hit)


def main():
    if not os.path.exists('pkl'):
        os.mkdir('pkl')
    if not os.path.exists(f'pkl{os.sep}sampled-300_labelled_data.pkl'):
        print('Pickle data file for tasks 1-3 does not exist, extracting data')
        extract_labelled()
        extract_unlabelled()

    preprocessed_data = task_1()
    task_2(preprocessed_data=preprocessed_data)
    task_3(preprocessed_data=preprocessed_data)


if __name__ == '__main__':
    main()
