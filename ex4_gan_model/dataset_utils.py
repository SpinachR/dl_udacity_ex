import os
import numpy
import collections
import scipy.io as sio
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

# print(sio.whosmat('SVHN/train_32x32.mat'))
Datasets = collections.namedtuple('Datasets', ['train', 'test'])


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_svhn(train_dir, num_classes=10, dtype=dtypes.float32, one_hot=False, reshape=False, seed=None):
    print('load trainset...')
    train_set = sio.loadmat(os.path.join(train_dir, 'train_32x32.mat'))
    train_x = numpy.transpose(train_set['X'], [3, 0, 1, 2])
    train_y = train_set['y']
    train_y[train_y == 10] = 0
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train_x, train_y, dtype=dtype, reshape=reshape, seed=seed)

    print('load testset...')
    test_set = sio.loadmat(os.path.join(train_dir, 'test_32x32.mat'))
    test_x = numpy.transpose(test_set['X'], [3, 0, 1, 2])
    test_y = test_set['y']
    test_y[test_y == 10] = 0
    if one_hot:
        test_y = dense_to_one_hot(test_y, num_classes)
    test = DataSet(test_x, test_y, dtype=dtype, reshape=reshape, seed=seed)

    # extra_set = sio.loadmat(os.path.join(train_dir, 'extra_32x32.mat'))
    # extra_x = numpy.transpose(extra_set['X'], [3, 0, 1, 2])
    # extra_y = extra_set['y']
    # extra_y[extra_y == 10] = 0
    # if one_hot:
    #     extra_y = dense_to_one_hot(extra_y, num_classes)
    # extra = DataSet(extra_x, extra_y, dtype=dtype, reshape=reshape, seed=seed)
    return Datasets(train=train, test=test)


# f = h5py.File('SVHN/svhn_format_2.hdf5', 'r+')


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 dtype=dtypes.float32,
                 reshape=True,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)

        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2] * images.shape[3])
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]
