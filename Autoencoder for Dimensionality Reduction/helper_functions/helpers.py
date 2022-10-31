import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

def print_layer_shapes(model):
    print('Encoder:')
    for layer in model.encoder.layers:
        print(layer._get_cell_name().replace('tf.keras.layers.', ''), layer.output_shape)
    print('Decoder:')
    for layer in model.decoder.layers:
        print(layer._get_cell_name().replace('tf.keras.layers.', ''), layer.output_shape)


def plot(data, title):
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.axis('off')
    columns = 11
    rows = 2
    for i in range(1, columns * rows + 1):
        if i == data.shape[0]:
            break
        img = data[i, :, :, 0]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='viridis')
        plt.axis('off')
    fig = plt.gcf()
    fig.suptitle(title, fontsize=40)
    plt.savefig(f'plots{os.sep}{title}.png')
    plt.show()


def pad_and_scale(data):
    data = np.pad(data, ((0, 0), (5, 6), (6, 6), (0, 0)), 'constant')
    data = data.reshape(data.shape[0], data.shape[1] * data.shape[2] * data.shape[3])
    data = data.reshape(data.shape[0], 320, 96, 1)
    return data

def pad_mcmc(data):
    data = np.pad(data, ((0, 0), (7, 7), (7, 7), (0, 0)), 'constant')
    data = data.reshape(data.shape[0], 64, 64, 1)

    return data

def neighborhood_hit(data, labels) -> float:
    # calculates the fraction of the k-nearest neighbors of a projected point P(x)
    # that have the same class label as P(x).
    neigh_hit = 0
    nbrs = NearestNeighbors(n_neighbors=17).fit(data)
    for idx, sample in enumerate(data):
        for neighbor in nbrs.kneighbors([sample])[1][0]:
            if labels[idx] == labels[neighbor]:
                neigh_hit += 1
    return neigh_hit / (len(data) * 17)