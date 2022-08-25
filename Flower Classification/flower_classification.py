import os
import tensorflow as tf


# supress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# get tf version and see if gpu is enabled
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
