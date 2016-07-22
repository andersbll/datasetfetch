import os
import numpy as np
import scipy.io
import logging

from .util import dataset_home, download, checksum, archive_extract, checkpoint


log = logging.getLogger(__name__)


CROPPED_URLS = [
    (
        'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
        'e6588cae42a1a5ab5efe608cc5cd3fb9aaffd674',
    ), (
        'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
        '29b312382ca6b9fba48d41a7b5c19ad9a5462b20'
    ), (
        'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat',
        'd7d93fbeec3a7cf69236a18015d56c7794ef7744'
    ),
]


class SVHN(object):
    '''
    The Street View House Numbers (SVHN) Dataset [1]

    http://ufldl.stanford.edu/housenumbers/

    References:
    [1]: Reading Digits in Natural Images with Unsupervised Feature Learning;
         Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu,
         Andrew Y. Ng; NIPS Workshop on Deep Learning and Unsupervised Feature
         Learning 2011.
    '''

    def __init__(self):
        self.name = 'svhn'
        self.n_classes = 10
        self.data_dir = os.path.join(dataset_home, self.name)
        self._install()
        self.arrays = self._load()

    def _install(self):
        checkpoint_file = os.path.join(self.data_dir, '__install_check')
        with checkpoint(checkpoint_file) as exists:
            if exists:
                return
            for url, sha1 in CROPPED_URLS:
                log.info('Downloading %s', url)
                filepath = download(url, self.data_dir)
                if sha1 != checksum(filepath, method='sha1'):
                    raise RuntimeError('Checksum mismatch for %s.' % _URL)

    def _load(self):
        dic = scipy.io.loadmat(os.path.join(self.data_dir, 'train_32x32.mat'))
        train_x = np.transpose(dic['X'], (3, 0, 1, 2))
        train_y = np.ravel(dic['y'])
        dic = scipy.io.loadmat(os.path.join(self.data_dir, 'test_32x32.mat'))
        test_x = np.transpose(dic['X'], (3, 0, 1, 2))
        test_y = np.ravel(dic['y'])
        dic = scipy.io.loadmat(os.path.join(self.data_dir, 'extra_32x32.mat'))
        extra_x = np.transpose(dic['X'], (3, 0, 1, 2))
        return train_x, train_y, test_x, test_y, extra_x
