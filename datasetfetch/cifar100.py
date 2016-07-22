import os
import pickle
import numpy as np
import logging

from .util import dataset_home, download, checksum, archive_extract, checkpoint


log = logging.getLogger(__name__)

_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
_SHA1 = 'eb9058c3a382ffc7106e4002c42a8d85'


class CIFAR100(object):
    '''
    The CIFAR-100 dataset [1]
    http://www.cs.toronto.edu/~kriz/cifar.html

    References:
    [1]: Learning Multiple Layers of Features from Tiny Images, Alex
         Krizhevsky, 2009.
    '''

    def __init__(self):
        self.name = 'cifar100'
        self.n_classes = 100
        self.n_test = 10000
        self.n_train = 50000
        self.img_shape = (3, 32, 32)
        self.data_dir = os.path.join(dataset_home, self.name)
        self._install()
        self._arrays = self._load()

    def arrays(self, flat=False):
        x_train, y_train, x_test, y_test = self._arrays
        if flat:
            x_train = np.reshape(x_train, (x_train.shape[0], -1))
            x_test = np.reshape(x_test, (x_test.shape[0], -1))
        return x_train, y_train, x_test, y_test

    def _install(self):
        checkpoint_file = os.path.join(self.data_dir, '__install_check')
        with checkpoint(checkpoint_file) as exists:
            if exists:
                return
            log.info('Downloading %s', _URL)
            filepath = download(_URL, self.data_dir)
            if _SHA1 != checksum(filepath, method='md5'):
                raise RuntimeError('Checksum mismatch for %s.' % _URL)

            log.info('Unpacking %s', filepath)
            archive_extract(filepath, self.data_dir)

    def _load(self):
        dirpath = os.path.join(self.data_dir, 'cifar-100-python')
        with open(os.path.join(dirpath, 'train'), 'rb') as f:
            dic = pickle.load(f)
            x_train = dic['data']
            y_train = np.array(dic['fine_labels'])
        with open(os.path.join(dirpath, 'test'), 'rb') as f:
            dic = pickle.load(f)
            x_test = dic['data']
            y_test = np.array(dic['fine_labels'])
        x_train = np.reshape(x_train, (self.n_train,) + self.img_shape)
        x_test = np.reshape(x_test, (self.n_test,) + self.img_shape)
        return x_train, y_train, x_test, y_test
