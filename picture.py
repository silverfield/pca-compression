__author__ = 'ferrard'

# ---------------------------------------------------------------
# Imports
# ---------------------------------------------------------------

import scipy as sp
from scipy import misc
from pca import Pca
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm

# ---------------------------------------------------------------
# Class
# ---------------------------------------------------------------


class CompressionInfo:
    def __init__(self, compressed_vectors=None, pca=None, shape=None):
        self.compressed_vectors = compressed_vectors
        self.pca = pca
        self.shape = shape

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump((self.compressed_vectors, self.pca, self.shape), f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            self.compressed_vectors, self.pca, self.shape = pickle.load(f)


class GreyScalePicture:
    # ---------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------

    def __init__(self):
        self.data = None

    # ---------------------------------------------------------------
    # Interface
    # ---------------------------------------------------------------

    def load_from_file(self, fname):
        self.data = misc.imread(fname, flatten=True)

    def write_to_file(self, png_fname):
        misc.imsave(png_fname, self.data)

    def load_from_data(self, data):
        self.data = data

    def get_vector(self):
        return self.data.flatten()

    def resize(self, new_bigger_dim):
        bigger = max(self.data.shape[0], self.data.shape[1])
        if new_bigger_dim > bigger:
            raise ValueError("Cannot resize to bigger dimension")

        ratio = new_bigger_dim/bigger

        new_data = sp.zeros((int(self.data.shape[0]*ratio), int(self.data.shape[1]*ratio)))
        for i in range(int(self.data.shape[0]*ratio)):
            for j in range(int(self.data.shape[1]*ratio)):
                new_data[i][j] = self.data[int(i/ratio)][int(j/ratio)]
        self.data = new_data

    def shape(self):
        return self.data.shape

    def print(self):
        print(self.data)

    def compress(self, compression):
        # compress using PCA
        k = int(self.data.shape[1]*compression)
        pca = Pca(self.data)
        compressed_vectors = [pca.project(vector, k) for vector in self.data]
        return CompressionInfo(compressed_vectors, pca, self.data.shape)

    def decompress(self, compression_info):
        pca = compression_info.pca
        compressed_vectors = compression_info.compressed_vectors
        shape = compression_info.shape

        # decompress using PCA
        self.data = sp.zeros(shape)
        for i in range(len(compressed_vectors)):
            decompressed_vector = pca.deproject(compressed_vectors[i])
            self.data[i, :] = decompressed_vector

    def show(self):
        plt.imshow(self.data, cmap=cm.Greys_r)
        plt.show()

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


def main():
    p = GreyScalePicture()
    p.load_from_file('others/small_lena.png')
    p.resize(128)
    ci = p.compress(0.7)
    p.decompress(ci)
    # p.comp(0.1)
    p.show()

if __name__ == '__main__':
    main()
