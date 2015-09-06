__author__ = 'ferrard'

# ---------------------------------------------------------------
# Imports
# ---------------------------------------------------------------

import numpy as np

# ---------------------------------------------------------------
# Class - normalization
# ---------------------------------------------------------------


class Normalization:
    """Provides methods to normalize data-set, and then based on the normalization parameters normalize or denormalize
    additional data-points
    """

    # ---------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------

    def __init__(self, train_x, train_y=None):
        """Normalizes the datasets on each column - so that the mean is 0 and std. dev is 1.

        Also remembers the normalization parameters for future normalization/denormalization.
        The data in the argument is left intact, new one is returned

        :param train_x: 2 dimensional ndarray (rows are individual X-values)
        :param train_y: list of Y-values
        :return: new, standardized data - X as ndarray and Y as a list
        """
        if train_y is not None:
            self.y_mean = np.mean(train_y)
            self.y_std = np.std(train_y)
        else:
            self.y_mean = None
            self.y_std = None
        self.x_means = list(np.mean(train_x, 0))
        self.x_stds = list(np.std(train_x, 0))

        self.norm_train_x = self.normalize_dataset_x(train_x)
        if train_y is not None:
            self.norm_train_y = self.normalize_dataset_y(train_y)
        else:
            self.norm_train_y = None

    # ---------------------------------------------------------------
    # Interface
    # ---------------------------------------------------------------

    def normalized_dataset(self):
        """Returns the normalized dataset which was used to construct this normalization"""
        if self.norm_train_y is None:
            return self.norm_train_x
        return self.norm_train_x, self.norm_train_y

    def normalize_x(self, x):
        """Returns normalized x, using saved normalization information

        :param x: a list/one dimensional ndarray
        :return: a list representing the normalized input
        """
        return [(float(x[i]) - self.x_means[i]) / self.x_stds[i] for i in range(len(x))]

    def normalize_dataset_x(self, data_x):
        """Returns the normalized dataset (the rows of data_x are normalized)"""
        norm_dataset_x = np.array([self.normalize_x(x) for x in data_x])
        return norm_dataset_x

    def denormalize_x(self, x):
        """Returns denormalized x, using saved normalization information

        :param x: a list/one dimensional ndarray
        :return: a list representing the denormalized input
        """
        return [float(x[i]) * self.x_stds[i] + self.x_means[i] for i in range(len(x))]

    def denormalize_dataset_x(self, norm_data_x):
        """Returns the denormalized dataset (the rows of norm_data_x are denormalized)"""
        denorm_dataset_x = np.array([self.denormalize_x(x) for x in norm_data_x])
        return denorm_dataset_x

    def normalize_y(self, y):
        """Returns normalized value of y using saved normalization information

        :param y: a single value
        :return: a normalized value of y
        """
        if self.y_mean is None or self.y_std is None:
            return None
        return (y - self.y_mean) / self.y_std

    def normalize_dataset_y(self, data_y):
        """Returns the normalized dataset (all y's of the vector data_y are normalized)"""
        norm_dataset_y = [self.normalize_y(y) for y in data_y]
        return norm_dataset_y

    def denormalize_y(self, y):
        """Returns denormalized value of y using saved normalization information

        :param y: a single value
        :return: a denormalized value of y
        """
        if self.y_mean is None or self.y_std is None:
            return None
        return y * self.y_std + self.y_mean

    def denormalize_dataset_y(self, norm_data_y):
        """Returns the denormalized dataset (all y's of the vector norm_data_y are denormalized)"""
        denorm_dataset_y = [self.denormalize_y(y) for y in norm_data_y]
        return denorm_dataset_y