import math
import random
import numpy as np
from tensorflow.keras.utils import Sequence


def _check_validity(data, labels, batch_size):
    if data.shape[0] != labels.shape[0]:
        raise ValueError('Args `x` and `y` must have the same length.')
    if data.shape[0] < 1:
        raise ValueError('Args `x` and `y` must not be empty.')
    if len(np.squeeze(labels).shape) != 2:
        raise ValueError(
            'Arg `y` must have a shape of (num_samples, num_classes). ' +
            'You can use `keras.utils.to_categorical` to convert a class ' +
            'vector to a binary class matrix.'
        )
    if batch_size < 1:
        raise ValueError('Arg `batch_size` must be a positive integer.')


def _initialization(data, labels, batch_size, categorical):
    num_samples = labels.shape[0]
    num_classes = labels.shape[1]
    classes_by_int = list(np.arange(num_classes))
    batch_data_shape = (batch_size, *data.shape[1:])
    batch_labels_shape = (batch_size, num_classes)\
        if categorical else (batch_size,)
    samples = [[] for _ in range(num_classes)]

    # Order samples according to their class
    for i in range(num_samples):
        samples[int(np.argmax(labels[i]))].append(data[i])
    for c, s in enumerate(samples):
        if len(s) < 1:
            raise ValueError('Class {} has no samples.'.format(c))

    return samples, num_classes, classes_by_int, batch_data_shape,\
           batch_labels_shape


def _balanced_batch(data, labels, samples, classes_per_batch,
                    samples_per_class, batch_size, num_classes, classes_by_int,
                    batch_data_shape, batch_labels_shape, shuffle, categorical,
                    rand):
    batch_data = np.ndarray(shape=batch_data_shape, dtype=data.dtype)
    batch_labels = np.zeros(shape=batch_labels_shape, dtype=labels.dtype)
    indexes = [0 for _ in range(num_classes)]
    random_class_pool = rand.sample(
        population=classes_by_int, k=classes_per_batch)
    for i in range(batch_size):
        random_class = rand.choice(random_class_pool)
        current_index = indexes[random_class]
        if current_index > samples_per_class - 2:
            random_class_pool.remove(random_class)
        indexes[random_class] = (current_index + 1)\
            % len(samples[random_class])
        if shuffle and current_index == 0:
            rand.shuffle(samples[random_class])
        batch_data[i] = samples[random_class][current_index]
        if categorical:
            batch_labels[i][random_class] = 1
        else:
            batch_labels[i] = random_class

    return batch_data, batch_labels


class BatchGenerator(Sequence):
    """A Keras-compatible generator to create balanced batches with
    `classes_per_batch` classes and `samples_per_class` samples for
    each class, respectively.

    This generator loops over its data indefinitely and yields balanced,
    shuffled batches. The generator will over-sample if necessary.

    Arguments:
    data (numpy.ndarray): Input data. Must have the same length as `y`.
    labels (numpy.ndarray): Target data. Must be a binary class matrix
        (i.e. shape `(num_samples, num_classes)`). You can use
        `keras.utils.to_categorical` to convert a class vector to a
        binary class matrix.
    params (object): Object that must contain `classes_per_batch` (int),
        `samples_per_class` (int), `shuffle` (bool).
    categorical (bool): (default=False) If true, generates binary class
        matrices (i.e. shape `(num_samples, num_classes)`) for batch
        targets. Otherwise, generates class vectors (i.e. shape
        `(num_samples,)`).
    seed: (default=None) Random seed.

    Returns:
        Keras-compatible generator yielding batches as `data, labels`.
    """
    def __init__(self, data, labels, params, categorical=False, seed=None):
        self.data = data
        self.labels = labels
        self.classes_per_batch = params.classes_per_batch
        self.samples_per_class = params.samples_per_class
        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.shuffle = params.shuffle
        self.categorical = categorical
        self.seed = seed
        self.rand = random.Random(self.seed)

        _check_validity(self.data, self.labels, self.batch_size)
        self.samples, self.num_classes, self.classes_by_int,\
            self.batch_data_shape, self.batch_labels_shape\
            = _initialization(self.data, self.labels, self.batch_size,
                              self.categorical)

    def __len__(self):
        """Number of batches in the sequence.

        Returns:
            The number of batches in the sequence.
        """

        return math.ceil(len(self.labels) / self.batch_size)

    def __getitem__(self, _):
        """Get one batch.

        Returns:
            batched_data, batched_labels: One batch.
        """
        return _balanced_batch(self.data, self.labels, self.samples,
                               self.classes_per_batch, self.samples_per_class,
                               self.batch_size, self.num_classes,
                               self.classes_by_int, self.batch_data_shape,
                               self.batch_labels_shape, self.shuffle,
                               self.categorical, self.rand)
