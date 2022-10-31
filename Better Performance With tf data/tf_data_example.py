import os
import time
import numpy as np
import tensorflow as tf

# make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Supress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
Best practice summary
Here is a summary of the best practices for designing performant TensorFlow input pipelines:

- Use the prefetch transformation to overlap the work of a producer and consumer
- Parallelize the data reading transformation using the interleave transformation
- Parallelize the map transformation by setting the num_parallel_calls argument
- Use the cache transformation to cache data in memory during the first epoch
- Vectorize user-defined functions passed in to the map transformation
- Reduce memory usage when applying the interleave, prefetch, and shuffle transformations
"""


class ArtificialDataset(tf.data.Dataset):
    def _generator(num_samples):
        # Opening the file
        time.sleep(0.03)

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)

            yield (sample_idx,)

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=tf.TensorSpec(shape=(1,), dtype=tf.int64),
            args=(num_samples,)
        )


def benchmark(dataset, num_epochs=2, approach=''):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    t = time.perf_counter() - start_time
    print(f'{approach} execution time: {round(t, 3)}')


fast_dataset = tf.data.Dataset.range(10000)


def fast_benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for _ in tf.data.Dataset.range(num_epochs):
        for _ in dataset:
            pass
    tf.print("Execution time:", time.perf_counter() - start_time)


def increment(x):
    return x + 1


def mapped_function(s):
    # Do some hard pre-processing
    tf.py_function(lambda: time.sleep(0.03), [], ())
    return s


def main():
    # The naive approach
    benchmark(ArtificialDataset(), approach='Naive')
    # Prefetch
    benchmark(ArtificialDataset().prefetch(tf.data.AUTOTUNE), approach='Prefetch')
    # Sequential interleave
    benchmark(tf.data.Dataset.range(2).interleave(lambda _: ArtificialDataset()), approach='Sequential Interleave')
    # Parallel interleave
    benchmark(
        tf.data.Dataset.range(2)
        .interleave(
            lambda _: ArtificialDataset(),
            num_parallel_calls=tf.data.AUTOTUNE
        ), approach='Parallel interleave'
    )

    # MAPPING
    # Sequential mapping
    benchmark(ArtificialDataset().map(mapped_function), approach='Sequential mapping')
    # Parallel Mapping
    benchmark(ArtificialDataset().map(mapped_function, num_parallel_calls=tf.data.AUTOTUNE),
              approach='Parallel Mapping')
    # Cashing
    benchmark(
        ArtificialDataset()
        .map(  # Apply time consuming operations before cache
            mapped_function
        ).cache(
        ),
        5, approach='Cashing'
    )

    # Scalar mapping
    fast_benchmark(
        fast_dataset
        # Apply function one item at a time
        .map(increment)
        # Batch
        .batch(256)
    )
    # Vectorized mapping
    fast_benchmark(
        fast_dataset
        .batch(256)
        # Apply function on a batch of items
        # The tf.Tensor.__add__ method already handle batches
        .map(increment)
    )


if __name__ == '__main__':
    main()
