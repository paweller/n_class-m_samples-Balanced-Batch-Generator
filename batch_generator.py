import numpy as np
from tensorflow.keras.utils import Sequence


def _check_validity(self):
    if self.data.shape[0] != self.labels.shape[0]:
        raise ValueError('Args `x` and `y` must have the same length.')
    if self.data.shape[0] < 1:
        raise ValueError('Args `x` and `y` must not be empty.')
    if len(self.labels.shape) != 2:
        raise ValueError(
            'Arg `y` must have a shape of (num_samples, num_classes). ' +
            'You can use `keras.utils.to_categorical` to convert a class ' +
            'vector to a binary class matrix.'
        )
    if self.batch_size < 1:
        raise ValueError('Arg `batch_size` must be a positive integer.')


def _initialization(self):
    num_samples = self.labels.shape[0]
    num_classes = self.labels.shape[1]
    classes_by_int = list(np.arange(num_classes))
    batch_data_shape = (self.batch_size, *self.data.shape[1:])
    batch_labels_shape = (self.batch_size, num_classes)\
        if self.categorical else (self.batch_size,)
    samples = [[] for _ in range(num_classes)]
    class_monitoring = np.ones(shape=num_classes, dtype=int)

    # Order samples according to their class
    for i in range(num_samples):
        samples[int(np.argmax(self.labels[i]))].append(self.data[i])
    for c, s in enumerate(samples):
        if len(s) < 1:
            raise ValueError('Class {} has no samples.'.format(c))

    return samples, num_classes, classes_by_int, batch_data_shape,\
           batch_labels_shape, class_monitoring


def _balanced_batch(self):
    batch_data = np.ndarray(shape=self.batch_data_shape,
                            dtype=self.data.dtype)
    batch_labels = np.zeros(shape=self.batch_labels_shape,
                            dtype=self.labels.dtype)
    class_sum = np.sum(self.class_monitoring)
    class_monitoring_probability = [class_sum / cm
                                    for cm in self.class_monitoring]
    class_sum = np.sum(class_monitoring_probability)
    class_monitoring_probability = [cm / class_sum
                                    for cm in class_monitoring_probability]
    random_class_pool = self.rng.choice(
        self.classes_by_int,
        size=self.classes_per_batch,
        replace=False,
        p=class_monitoring_probability
    )
    for class_idx in random_class_pool:
        self.class_monitoring[class_idx] += 1

    indexes = [0 for _ in range(self.num_classes)]
    for i in range(self.batch_size):
        random_class = self.rng.choice(random_class_pool, replace=True)
        current_index = indexes[random_class]
        if current_index > self.samples_per_class - 2:
            random_class_pool = np.delete(random_class_pool,
                np.argwhere(random_class_pool == random_class))
        indexes[random_class] = (current_index + 1)\
            % len(self.samples[random_class])
        if self.shuffle and current_index == 0:
            self.rng.shuffle(self.samples[random_class])
        batch_data[i] = self.samples[random_class][current_index]
        if self.categorical:
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
        self.rng = np.random.default_rng(self.seed)

        _check_validity(self)
        self.samples, self.num_classes, self.classes_by_int,\
            self.batch_data_shape, self.batch_labels_shape,\
            self.class_monitoring = _initialization(self)

    def __len__(self):
        """Number of batches in the sequence.

        Returns:
            The number of batches in the sequence.
        """

        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, _):
        """Get one batch.

        Returns:
            batch_data, batch_labels: One batch.
        """
        return _balanced_batch(self)
