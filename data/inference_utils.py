from __future__ import print_function
from __future__ import division

import numpy as np
import h5py as hf
import sys, os
import matplotlib.pyplot as plt

def imshow_loop(loopsites, path, name):
    imgmap = np.zeros(1024)
    imgmap[loopsites] = 1
    plt.imshow(imgmap.reshape(32,32), interpolation='None', cmap='plasma')
    plt.savefig(os.path.join(path, name) + '.png')
    plt.title(name)
    plt.clf()

def imshow_probmap(probs, path, name):
    plt.imshow(probs.reshape(32,32), interpolation='None', cmap='plasma')
    plt.colorbar()
    plt.savefig(os.path.join(path, name) + '.png')
    plt.title(name)
    plt.clf()

class InferenceData(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        ds = hf.File(self.dataset_path, 'r')
        states = ds['icestates'][:]
        self.images = states
        ds.close()

        # Create labels

        # Login dataset spec
        self._num_of_samples = self.images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        print ('Read dataset with {} images.'.format(self._num_of_samples))

    @property
    def num_samples(self):
        return self._num_of_samples

    def next_batch(self, batch_size, shuffle=False):
        start = self._index_in_epoch
        # Shuffle the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_of_samples)
            #np.random.shuffle(perm0)
            self.images = self.images[perm0]
                # Finsh the epoch
        if start + batch_size > self._num_of_samples:
            self._epochs_completed += 1
            rest_num_samples = self._num_of_samples - start
            images_rest_part = self.images[start:self._num_of_samples]
            # Shuffle
            if shuffle:
                perm = np.arange(self._num_of_samples)
                #np.random.shuffle(perm)
                self.images = self.images[perm]
                # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_samples
            end = self._index_in_epoch
            images_new_part = self.images[start:end]

            return np.concatenate((images_rest_part, images_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.images[start:end]