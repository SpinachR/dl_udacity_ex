import os
import numpy
import collections
import scipy.io as sio
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import pickle
from scipy import linalg
import tensorflow as tf

Datasets = collections.namedtuple('Datasets', ['train', 'test'])


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
    
    
def load_cifar9(dir, num_classes=9, dtype=dtypes.uint8, one_hot=False, reshape=False, seed=None):
    print('load cifar9 trainset 0-1...')
    train_dir = os.path.join(dir, 'cifar9_train0_1.pkl')
    with open(train_dir, 'rb') as f:
        train = pickle.load(f)

    train_y = train['y']
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train['X'], train_y, dtype=dtype, reshape=reshape, seed=seed)

    print('load cifar9 testset 0-1...')
    test_dir = os.path.join(dir, 'cifar9_test0_1.pkl')
    with open(test_dir, 'rb') as f:
        test = pickle.load(f)

    test_y = test['y']
    if one_hot:
        test_y = dense_to_one_hot(test_y, num_classes)
    test = DataSet(test['X'], test_y, dtype=dtype, reshape=reshape, seed=seed)
    return Datasets(train=train, test=test)
    
    
def load_stl9(dir, num_classes=9, dtype=dtypes.uint8, one_hot=False, reshape=False, seed=None):
    print('load stl9 trainset 0-1...')
    train_dir = os.path.join(dir, 'stl9_train_skidown0_1.pkl')
    with open(train_dir, 'rb') as f:
        train = pickle.load(f)

    train_y = train['y']
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train['X'], train_y, dtype=dtype, reshape=reshape, seed=seed)

    print('load stl9 testset 0-1...')
    test_dir = os.path.join(dir, 'stl9_test_skidown0_1.pkl')
    with open(test_dir, 'rb') as f:
        test = pickle.load(f)

    test_y = test['y']
    if one_hot:
        test_y = dense_to_one_hot(test_y, num_classes)
    test = DataSet(test['X'], test_y, dtype=dtype, reshape=reshape, seed=seed)
    return Datasets(train=train, test=test)    


def load_mnist(dir, num_classes=10, dtype=dtypes.float32, one_hot=False, reshape=False, seed=None):
    print('load mnist trainset...')
    train_dir = os.path.join(dir, 'train32.pkl')
    with open(train_dir, 'rb') as f:
        train = pickle.load(f)

    train_y = train['y']
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train['X'], train_y, dtype=dtype, reshape=reshape, seed=seed)

    print('load mnist testset...')
    test_dir = os.path.join(dir, 'test32.pkl')
    with open(test_dir, 'rb') as f:
        test = pickle.load(f)

    test_y = test['y']
    if one_hot:
        test_y = dense_to_one_hot(test_y, num_classes)
    test = DataSet(test['X'], test_y, dtype=dtype, reshape=reshape, seed=seed)
    return Datasets(train=train, test=test)

    
def load_mnist28(dir, num_classes=10, dtype=dtypes.float32, one_hot=False, reshape=False, seed=None):
    print('load mnist trainset...')
    train_dir = os.path.join(dir, 'train_mnist_28.pkl')
    with open(train_dir, 'rb') as f:
        train = pickle.load(f)

    train_y = train['y']
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train['X'], train_y, dtype=dtype, reshape=reshape, seed=seed)

    print('load mnist testset...')
    test_dir = os.path.join(dir, 'test_mnist_28.pkl')
    with open(test_dir, 'rb') as f:
        test = pickle.load(f)

    test_y = test['y']
    if one_hot:
        test_y = dense_to_one_hot(test_y, num_classes)
    test = DataSet(test['X'], test_y, dtype=dtype, reshape=reshape, seed=seed)
    return Datasets(train=train, test=test)

    

def load_mnist_m(dir, num_classes=10, dtype=dtypes.float32, one_hot=False, reshape=False, seed=None):
    print('load mnist_m trainset...')
    train_dir = os.path.join(dir, 'm_train32.pkl')
    with open(train_dir, 'rb') as f:
        train = pickle.load(f)

    train_y = train['y']
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train['X'], train_y, dtype=dtype, reshape=reshape, seed=seed)

    print('load mnist_m testset...')
    test_dir = os.path.join(dir, 'm_test32.pkl')
    with open(test_dir, 'rb') as f:
        test = pickle.load(f)

    test_y = test['y']
    if one_hot:
        test_y = dense_to_one_hot(test_y, num_classes)
    test = DataSet(test['X'], test_y, dtype=dtype, reshape=reshape, seed=seed)
    return Datasets(train=train, test=test)

    
def load_mnist_m28(dir, num_classes=10, dtype=dtypes.float32, one_hot=False, reshape=False, seed=None):
    print('load mnist_m trainset...')
    train_dir = os.path.join(dir, 'train_mnist_m_28.pkl')
    with open(train_dir, 'rb') as f:
        train = pickle.load(f)

    train_y = train['y']
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train['X'], train_y, dtype=dtype, reshape=reshape, seed=seed)

    print('load mnist_m testset...')
    test_dir = os.path.join(dir, 'test_mnist_m_28.pkl')
    with open(test_dir, 'rb') as f:
        test = pickle.load(f)

    test_y = test['y']
    if one_hot:
        test_y = dense_to_one_hot(test_y, num_classes)
    test = DataSet(test['X'], test_y, dtype=dtype, reshape=reshape, seed=seed)
    return Datasets(train=train, test=test)

def load_usps(dir, num_classes=10, dtype=dtypes.float32, one_hot=False, reshape=False, seed=None):
    print('load usps trainset...')
    train_dir = os.path.join(dir, 'train_usps32_bicubic.pkl')
    with open(train_dir, 'rb') as f:
        train = pickle.load(f)

    train_y = train['y']
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train['X'], train_y, dtype=dtype, reshape=reshape, seed=seed)

    print('load usps testset...')
    test_dir = os.path.join(dir, 'test_usps32_bicubic.pkl')
    with open(test_dir, 'rb') as f:
        test = pickle.load(f)

    test_y = test['y']
    if one_hot:
        test_y = dense_to_one_hot(test_y, num_classes)
    test = DataSet(test['X'], test_y, dtype=dtype, reshape=reshape, seed=seed)
    return Datasets(train=train, test=test)
    
    
def load_usps_01(dir, num_classes=10, dtype=dtypes.uint8, one_hot=False, reshape=False, seed=None):
    print('load usps trainset...')
    train_dir = os.path.join(dir, 'train_usps32_bilinear_01.pkl')
    with open(train_dir, 'rb') as f:
        train = pickle.load(f)

    train_y = train['y']
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train['X'], train_y, dtype=dtype, reshape=reshape, seed=seed)

    print('load usps testset...')
    test_dir = os.path.join(dir, 'test_usps32_bilinear_01.pkl')
    with open(test_dir, 'rb') as f:
        test = pickle.load(f)

    test_y = test['y']
    if one_hot:
        test_y = dense_to_one_hot(test_y, num_classes)
    test = DataSet(test['X'], test_y, dtype=dtype, reshape=reshape, seed=seed)
    return Datasets(train=train, test=test)
    

def load_svhn(train_dir, num_classes=10, dtype=dtypes.float32, one_hot=False, reshape=False, seed=None):
    print('load svhn trainset...')
    train_set = sio.loadmat(os.path.join(train_dir, 'train_32x32.mat'))
    train_x = numpy.transpose(train_set['X'], [3, 0, 1, 2])
    train_y = train_set['y']
    train_y = train_y.reshape(-1)
    train_y[train_y == 10] = 0
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train_x, train_y, dtype=dtype, reshape=reshape, seed=seed)

    print('load svhn testset...')
    test_set = sio.loadmat(os.path.join(train_dir, 'test_32x32.mat'))
    test_x = numpy.transpose(test_set['X'], [3, 0, 1, 2])
    test_y = test_set['y']
    test_y = test_y.reshape(-1)
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
    
    

def load_gtsrb(data_dir, num_classes=43, dtype=dtypes.uint8, one_hot=False, reshape=False, seed=None):
    print('load gtsrb trainset...')
    train_dir = os.path.join(data_dir, 'train_gtrsb32_pil.pkl')
    with open(train_dir, 'rb') as f:
        train = pickle.load(f)

    train_y = train['y']
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
        
    num_examples = len(train['X'])
    print('number of total gtsrb: ', num_examples)
    random_index = numpy.random.permutation(num_examples)
    X = train['X']
    trainx = X[random_index[:31367], :, :, :]
    trainy = train_y[random_index[:31367]]
    
    # train = DataSet(train['X'], train_y, dtype=dtype, reshape=reshape, seed=seed)
    train = DataSet(trainx, trainy, dtype=dtype, reshape=reshape, seed=seed)
    
    # test
    testx = X[random_index[31367:], :, :, :]
    testy = train_y[random_index[31367:]]
    print('number of test: ', len(testx))
    test = DataSet(testx, testy, dtype=dtype, reshape=reshape, seed=seed)
    # print('load gtsrb testset...')
    # test_dir = os.path.join(data_dir, 'test_gtrsb32_pil.pkl')
    # with open(test_dir, 'rb') as f:
        # test = pickle.load(f)

    # test_y = test['y']
    # if one_hot:
        # test_y = dense_to_one_hot(test_y, num_classes)
    # test = DataSet(test['X'], test_y, dtype=dtype, reshape=reshape, seed=seed)
    return Datasets(train=train, test=test)
    
    
    
def load_synsign(data_dir, num_classes=43, dtype=dtypes.uint8, one_hot=False, reshape=False, seed=None):
    print('load syn. sign trainset...')
    train_dir = os.path.join(data_dir, 'train_synsign32_pil.pkl')
    with open(train_dir, 'rb') as f:
        train = pickle.load(f)

    train_y = train['y']
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train['X'], train_y, dtype=dtype, reshape=reshape, seed=seed)
    return train
    
    
def load_syndigit(data_dir, num_classes=10, dtype=dtypes.uint8, one_hot=False, reshape=False, seed=None):
    print('load syn. digit trainset...')
    train_dir = data_dir
    with open(train_dir, 'rb') as f:
        train = pickle.load(f)

    num_examples = len(train['X'])
    idx = numpy.random.permutation(num_examples)
    train_x = train['X']
    train_y = train['y']

    train_x = train_x[idx]
    train_y = train_y[idx]
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train_x, train_y, dtype=dtype, reshape=reshape, seed=seed)
    return train

    


def load_cifar(data_dir, dtype=dtypes.float32, seed=None):
    print('Loading cifar10...')
    with open(data_dir+'/data_batch_1', 'rb') as fo:
        train1 = pickle.load(fo, encoding='latin1')
        train1X = numpy.reshape(train1['data'], [10000, 3, 32, 32])
        train1Y = numpy.array(train1['labels'])
    with open(data_dir+'/data_batch_2', 'rb') as fo:
        train2 = pickle.load(fo, encoding='latin1')
        train2X = numpy.reshape(train2['data'], [10000, 3, 32, 32])
        train2Y = numpy.array(train2['labels'])

    with open(data_dir+'/data_batch_3', 'rb') as fo:
        train3 = pickle.load(fo, encoding='latin1')
        train3X = numpy.reshape(train3['data'], [10000, 3, 32, 32])
        train3Y = numpy.array(train3['labels'])

    with open(data_dir+'/data_batch_4', 'rb') as fo:
        train4 = pickle.load(fo, encoding='latin1')
        train4X = numpy.reshape(train4['data'], [10000, 3, 32, 32])
        train4Y = numpy.array(train4['labels'])

    with open(data_dir+'/data_batch_5', 'rb') as fo:
        train5 = pickle.load(fo, encoding='latin1')
        train5X = numpy.reshape(train5['data'], [10000, 3, 32, 32])
        train5Y = numpy.array(train5['labels'])

    with open(data_dir+'/test_batch', 'rb') as fo:
        test = pickle.load(fo, encoding='latin1')
        testX = numpy.reshape(test['data'], [10000, 3, 32, 32])
        testY = numpy.array(test['labels'])


    trX = numpy.concatenate((train1X,train2X,train3X,train4X,train5X),axis=0).astype(numpy.float)
    trX = numpy.swapaxes(trX, 1, 3)
    trX = numpy.swapaxes(trX, 1, 2)
    trY = numpy.concatenate((train1Y,train2Y,train3Y,train4Y,train5Y),axis=0).astype(numpy.int)
    train = DataSet(trX, trY, dtype=dtype, reshape=False, seed=seed)

    teX = numpy.swapaxes(testX, 1, 3)
    teX = numpy.swapaxes(teX, 1, 2).astype(numpy.float)
    teY = testY.astype(numpy.int)
    test = DataSet(teX, teY, dtype=dtype, reshape=False, seed=seed)

    print('Done.')
    return Datasets(train=train, test=test)
    
def load_A(data_dir, num_classes=31, dtype=dtypes.uint8, one_hot=False, reshape=False, seed=None):
    print('load %s...' % data_dir)
    with open(data_dir, 'rb') as f:
        train = pickle.load(f)

    train_y = train['y']
    if one_hot:
        train_y = dense_to_one_hot(train_y, num_classes)
    train = DataSet(train['X'], train_y, dtype=dtype, reshape=reshape, seed=seed)
    return train

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


#########################################################################################################
def ZCA(data, reg=1e-6):
    mean = numpy.mean(data, axis=0)
    mdata = data - mean
    sigma = numpy.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = numpy.dot(numpy.dot(U, numpy.diag(1 / numpy.sqrt(S) + reg)), U.T)
    whiten = numpy.dot(data - mean, components.T)
    return components, mean, whiten


def read_tfrecords(tfrecordNames, size, num_epochs=None):
    print("filenames in queue:", tfrecordNames)
    filename_queue = tf.train.string_input_producer(tfrecordNames, num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, size)
    label = features['label']
    return img, label


def generate_batch(
        example,
        min_queue_examples,
        batch_size, shuffle):
    """
    Arg:
        list of tensors.
    """
    num_preprocess_threads = 1

    if shuffle:
        ret = tf.train.shuffle_batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        ret = tf.train.batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            allow_smaller_final_batch=True,
            capacity=min_queue_examples + 3 * batch_size)

    return ret


def transform(image, is_crop, is_flip):
    image = tf.reshape(image, [32, 32, 3])
    if is_crop or is_flip:
        print("augmentation")
        if is_crop:
            image = tf.pad(image, [[2, 2], [2, 2], [0, 0]])
            image = tf.random_crop(image, [32, 32, 3])
        if is_flip:
            image = tf.image.random_flip_left_right(image)
    return image


def input(fileNames, img_size, batch_size, num_examples, shuffle=True, num_epochs=None):
    img, label = read_tfrecords(fileNames, img_size, num_epochs)
    # img = transform(img)
    return generate_batch([img, label], num_examples, batch_size, shuffle)
