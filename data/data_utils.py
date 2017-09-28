import numpy as np
import h5py as hf
import sys, os

def get_filelist(source_idx, prefix='loopstate', dirname='loops'):
    files = os.listdir(dirname)
    filelist = []
    for f in files:
        if str.startswith(f, prefix):
            fname = f.rstrip('.npy')
            trans = fname.split('_')[-1]
            from_idx, to_idx = trans.split('-')
            if (int(from_idx) == source_idx):
                #print ('read file: {}'.format(fname))
                filelist.append(f)
    return filelist

def read_filelist(filelist, dirname='loops'):
    loops = []
    for f in filelist:
        loops.extend(np.load('/'.join([dirname, f]))[0])
    return loops

def get_mask(sequences):
    return [0 if s == 0 else 1 for s in sequences[1:]]

def get_batch_mask(batch_seq):
    batch_mask = []
    for seq in batch_seq:
        batch_mask.append(get_mask(seq))
    return np.array(batch_mask)

def read_markovchain_dataset(dataset_path):
    dataset = hf.File(dataset_path, 'r')
    num_states = len(dataset.keys()) / 2
    print ('{} dataset contains {} initial states'.format(dataset_path, num_states))
    states, loops = [], []
    for i in range(num_states):
        states.extend(dataset['MC_{}_states'.format(i)][:])
        loops.extend(dataset['MC_{}_loops'.format(i)][:])
    dataset.close()
    return np.array(states), np.array(loops)

def read_iceloop_dataset(dataset_path):
    dataset = hf.File(dataset_path, 'r')
    states, loops = [], []
    states = dataset['STATES'][:]
    loops = dataset['LOOPS'][:]
    return np.array(states), np.array(loops)

class DataReader(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        # TODO: change to IceLoop format
        #states, loops = read_markovchain_dataset(self.dataset_path)
        states, loops = read_iceloop_dataset(self.dataset_path)
        masks = get_batch_mask(loops)

        self.images = states
        self.sequences = loops
        self.masks = masks

        # Login dataset spec
        self._num_of_samples = self.images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
        print ('Load dataset with {} images from {}'.format(self._num_of_samples, self.dataset_path))

    @property
    def num_samples(self):
        return self._num_of_samples

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_of_samples)
            np.random.shuffle(perm0)
            self.images = self.images[perm0]
            self.sequences = self.sequences[perm0]
            self.masks = self.masks[perm0]
                # Finsh the epoch
        if start + batch_size > self._num_of_samples:
            self._epochs_completed += 1
            rest_num_samples = self._num_of_samples - start
            images_rest_part = self.images[start:self._num_of_samples]
            sequences_rest_part = self.sequences[start:self._num_of_samples]
            masks_rest_part = self.masks[start:self._num_of_samples]
            # Shuffle
            if shuffle:
                perm = np.arange(self._num_of_samples)
                np.random.shuffle(perm)
                self.images = self.images[perm]
                self.sequences = self.sequences[perm]
                self.masks = self.masks[perm]
                # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_samples
            end = self._index_in_epoch
            images_new_part = self.images[start:end]
            sequences_new_part = self.sequences[start:end]
            masks_new_part = self.masks[start:end]

            # Return input & target sequences (an expansive method)
            sequences = np.concatenate((sequences_rest_part, sequences_new_part), axis=0)
            input_sequences =[]
            target_sequences =[]
            for seq in sequences:
                input_sequences.append([s for s in seq[:-1]])
                target_sequences.append([s for s in seq[1:]])
            input_sequences = np.array(input_sequences)
            target_sequences = np.array(target_sequences)

            return np.concatenate((images_rest_part, images_new_part), axis=0), \
                input_sequences, target_sequences, \
                np.concatenate((masks_rest_part, masks_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            input_sequences =[]
            target_sequences =[]
            for seq in self.sequences[start:end]:
                input_sequences.append([s for s in seq[:-1]])
                target_sequences.append([s for s in seq[1:]])
            input_sequences = np.array(input_sequences)
            target_sequences = np.array(target_sequences)
            return self.images[start:end], input_sequences, target_sequences, self.masks[start:end]