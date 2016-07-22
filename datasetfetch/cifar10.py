import os
import pickle
import numpy as np
import logging

from .util import dataset_home, download, checksum, archive_extract, checkpoint


log = logging.getLogger(__name__)

_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_SHA1 = '874905e36347c8536514d0a26261acf3bff89bc7'


class CIFAR10(object):
    '''
    The CIFAR-10 dataset [1]
    http://www.cs.toronto.edu/~kriz/cifar.html

    References:
    [1]: Learning Multiple Layers of Features from Tiny Images, Alex
         Krizhevsky, 2009.
    '''

    def __init__(self):
        self.name = 'cifar10'
        self.n_classes = 10
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
            if _SHA1 != checksum(filepath, method='sha1'):
                raise RuntimeError('Checksum mismatch for %s.' % _URL)

            log.info('Unpacking %s', filepath)
            archive_extract(filepath, self.data_dir)

    def _load(self):
        dirpath = os.path.join(self.data_dir, 'cifar-10-batches-py')
        filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                     'data_batch_4', 'data_batch_5', 'test_batch']
        x = []
        y = []
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'rb') as f:
                dic = pickle.load(f)
                x.append(dic['data'])
                y.append(dic['labels'])
        x_train = np.vstack(x[:5])
        y_train = np.hstack(y[:5])
        x_test = np.array(x[5])
        y_test = np.array(y[5])
        x_train = np.reshape(x_train, (self.n_train,) + self.img_shape)
        x_test = np.reshape(x_test, (self.n_test,) + self.img_shape)
        return x_train, y_train, x_test, y_test
