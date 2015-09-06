# ---------------------------------------------------------------
# Imports
# ---------------------------------------------------------------

import normalization
import scipy as sp
from scipy import linalg
import csv

# ---------------------------------------------------------------
# Class - Pca
# ---------------------------------------------------------------


class Pca:
    """Principal component analysis on a matrix representing the data set (one item = one row)

    The class serves the purpose of projecting data to lower dimensions given by top principal components
    (and back, however with some loss) to obtain compression/dimensionality reduction
    """
    # ---------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------

    def __init__(self, data):
        # normalize
        self._normalization = normalization.Normalization(data)
        normalized_data = self._normalization.normalized_dataset()
        # normalized_data = data

        # find covariance matrix
        data_matrix = sp.matrix(normalized_data)
        m = data_matrix.shape[0]
        covariance_matrix = data_matrix.transpose()*data_matrix
        covariance_matrix /= m

        # find principal components
        eig_decomp = linalg.eigh(covariance_matrix)  # sorted by eig. values (ascending - we want descending)
        self._n = len(eig_decomp[0])
        self._pcas = sp.zeros((self._n, self._n))  # one row will be one princ. comp. (starting from most important PC)
        for i in range(self._n):
            self._pcas[i, :] = eig_decomp[1][:, self._n - i - 1]

        self._eig_vals = list(eig_decomp[0])
        self._eig_vals.reverse()

    # ---------------------------------------------------------------
    # Interface
    # ---------------------------------------------------------------

    @property
    def pcas(self):
        """Returns the principal components (from most important to least)"""
        return self._pcas

    @property
    def eig_vals(self):
        """Returns the eigen values for the the principal components"""
        return self._eig_vals

    @property
    def n(self):
        """Returns the dimensionality of the data for which we run PCA"""
        return self._n

    def project(self, vector, k):
        """Projects the vector to the space given by top k principal components

        :returns a list of numbers representing vector in the space given by top K principal components
        """
        # normalize the vector
        v = self._normalization.normalize_x(vector)

        # project it
        v = sp.array(v)
        dot_product = lambda pca: sum(pca[j]*v[j] for j in range(len(v)))
        return [dot_product(self.pcas[i]) for i in range(k)]

    def deproject(self, vector):
        """De-projects the vector back to the original space.

        We assume the vector comes from the top k principal components space, where k = dim(v)

        :returns a list of numbers representing the vector in the original space
        """
        # deproject the vector
        v = list(vector)
        result = sp.zeros(self._n)
        for i in range(len(v)):
            result += self._pcas[i]*v[i]

        # denormalize it
        result = self._normalization.denormalize_x(list(result))
        return list(result)

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


def main():
    rows = []
    delimiter = ','
    with open('datasets/housing.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            rows.append([float(r) for r in row])
    data = sp.array(rows)

    pca = Pca(data)
    K = 10
    before_compression = data[5, :]
    compressed = pca.project(before_compression, K)

    after_decompression = pca.deproject(compressed)

    print("Before compression: \n" + str(before_compression))
    print("Compressed: \n" + str(compressed))
    print("After decompression: \n" + str(after_decompression))

    print("Difference: \n" + str(list(after_decompression - before_compression)))

    # print(data)

if __name__ == '__main__':
    main()
