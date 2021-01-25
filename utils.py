import json
import numpy as np


def reshape_data(data):
    if data.shape[-1] == data.shape[-2]:
        # Data without channel information (i.e. BW images).
        if len(data.shape) == 4:
            # Input data of shape (classes, class_samples, h, w).
            # Reshape and expand to (total_samples, h, w, c).
            data = np.reshape(data, (-1, *data.shape[2:]))
            data = np.expand_dims(data, -1)
        elif len(data.shape) < 3 or len(data.shape) > 4:
            raise ValueError('Unsupported shape {} of input data.'
                             .format(data.shape))
    elif data.shape[-1] == 1 or data.shape[-1] == 3:
        # Data with channel information (i.e. BW or RGB images).
        if len(data.shape) == 5:
            # Input data of shape (classes, class_samples, h, w, c).
            # Reshape to (total_samples, h, w, c).
            data = np.reshape(data, (-1, *data.shape[2:]))
        elif len(data.shape) < 4 or len(data.shape) > 5:
            raise ValueError('Unsupported shape {} of input data.'
                             .format(data.shape))
    else:
        raise ValueError('Unknown data type.')

    return data


class Params():
    """Load/write hyperparameters from/to a json file.

    Example code:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5
    ```
    """

    def __init__(self, pd):
        self.load(pd)

    def save(self, pd):
        with open(pd, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, pd):
        with open(pd) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to an instance of Params."""
        return self.__dict__
