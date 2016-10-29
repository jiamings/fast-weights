import numpy as np
import collections
try:
    import cPickle as pickle
except ImportError:
    import pickle


Datasets = collections.namedtuple('Datasets', ['train', 'val', 'test'])


class Dataset(object):
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._epoch_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self.x.shape[0]
        self.perm = np.random.permutation(np.arange(self._num_examples))

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        assert batch_size <= self._num_examples
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch >= self.num_examples:
            self._epoch_completed += 1
            np.random.shuffle(self.perm)
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._x[self.perm[start:end]], self._y[self.perm[start:end]]


def read_data(data_path='associative-retrieval.pkl'):
    with open(data_path, 'rb') as f:
        d = pickle.load(f)
    x_train = d['x_train']
    x_val = d['x_val']
    x_test = d['x_test']
    y_train = d['y_train']
    y_val = d['y_val']
    y_test = d['y_test']
    train = Dataset(x_train, y_train)
    test = Dataset(x_test, y_test)
    val = Dataset(x_val, y_val)
    return Datasets(train=train, val=val, test=test)

