# n_class-m_samples-Balanced-Batch-Generator: Keras-compatible generator to create batches with n classes and m samples per class.

[![MIT license](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains an algorithm to create balanced batches containing `classes_per_batch` batches and `samples_per_class` samples for each class, respectively. If necessary, the algorithm will over-sample. If `shuffle` is set to `True` generated batches are shuffled. The generator supports SISO models.

The generator is compatible with Keras models' [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) method.

## API

```python
generator = BatchGenerator(
    data=data,
    labels=labels,
    params=params,
    categorical=False,
    seed=None
)
```

Arguments:
- `data` *(numpy.ndarray)*: Input data.
- `labels` *(numpy.ndarray)*: Data's labels.
- `params` *(object)*: Object of a class. Must include parameters `classes_per_batch` *(int)*, `samples_per_class` *(int)* and `shuffle` *(int)*.
- `categorical` *(bool)(optional)(default=False)*: If true, the generator yields binary class matrices. Otherwise, it yields class vectors.
- `seed` *(optional)(default=None)*: Random seed.

Returns:
- A Keras-compatible generator yielding batches as `batch_data, batch_lables`.

## Dummy code example

```python
from tensorflow.keras.utils import to_categorical

from utils import Params, reshape_data
from batch_generator import BatchGenerator

# Load the parameters
params_pd = './params.json'
params = Params(params_pd)

# Load data and labels
train_data = ...
train_labels = ...   # shape (num_labels,)
valid_data = ...
valid_labels = ...   # shape (num_labels,)

# Optional step: To make sure data is formatted correctly, the
# `reshape_data` function from the `utils.py` file can be used.
train_data = reshape_data(train_data)
valid_data = reshape_data(valid_data)

# Create generator objects for training and validation
train_generator = BatchGenerator(
    data=train_data,
    labels=to_categorical(train_labels),
    params=params
)

valid_generator = BatchGenerator(
    data=valid_data,
    labels=to_categorical(valid_labels),
    params=params
)

# Create, compile and fit a sequential model
model = ...

siamese_net_model.compile(
    optimizer=...,
    loss=...
)

history = siamese_net_model.fit(
    x=train_generator,
    validation_data=valid_generator,
    epochs=params.num_epochs
)
```

## Sources

- This repository builds on soroushj implementation of a [keras-balanced-batch-generator: A Keras-compatible generator for creating balanced batches](https://github.com/soroushj/keras-balanced-batch-generator). Go over to his GitHub page and check out his work.
