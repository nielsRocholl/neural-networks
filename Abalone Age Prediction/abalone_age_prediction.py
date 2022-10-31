import keras.layers
import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


def main():
    abalone_train = pd.read_csv(
        "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
        names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
               "Viscera weight", "Shell weight", "Age"])

    abalone_train.head()


    abalone_features = abalone_train.copy()
    abalone_labels = abalone_features.drop('Age')

    abalone_features = np.array(abalone_features)


if __name__ == '__main__':
    main()