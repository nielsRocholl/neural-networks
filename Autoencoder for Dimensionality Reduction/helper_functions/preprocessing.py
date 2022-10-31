import os

from matplotlib import pyplot as plt

def preprocess(data, visualize, cropping, mcmc):
    # reshape, visualize, normalize, scale

    # create dir for visualizations
    if not os.path.exists(f'plots{os.sep}'):
        os.mkdir(f'plots{os.sep}')

    if visualize:
        # visualize all data
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
        plt.suptitle('Original data')
        if mcmc:
            plt.savefig(f'plots{os.sep}original_data_mcmc.png')
        else:
            plt.savefig(f'plots{os.sep}original_data.png')
        plt.show()

    # normalize data: subtract mean and std dev
    normalized_data = data


    # normalize to zero mean and unit variance
    data_mean = normalized_data.mean()
    data_std = normalized_data.std()
    normalized_data = (normalized_data - normalized_data.mean()) / normalized_data.std()

    if visualize:
        # visualize all normalized and scaled data
        fig, ax = plt.subplots(figsize=(20, 15))
        ax.axis('off')
        columns = 11
        rows = 2
        for i in range(1, columns * rows + 1):
            if i == normalized_data.shape[0]:
                break
            img = normalized_data[i, :, :, 0]
            fig.add_subplot(rows, columns, i)
            plt.imshow(img, cmap='viridis', vmin=normalized_data.min(), vmax=normalized_data.max())
            plt.axis('off')
        fig = plt.gcf()
        plt.suptitle(f'Normalized data')
        if mcmc:
            plt.savefig(f'plots{os.sep}original_data_mcmc.png')
        else:
            plt.savefig(f'plots{os.sep}original_data.png')
        plt.show()

    # crop the input to remove cylinder feature;
    if cropping:
        cut_value = int(0.3 * normalized_data.shape[1])
        normalized_data = normalized_data[:, cut_value:, :, :]
        # print("normalized_cropped_data: ", normalized_cropped_data.shape)

    return normalized_data, data_mean, data_std
