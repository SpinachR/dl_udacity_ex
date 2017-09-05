import scipy.io as sio
import numpy
import os
import matplotlib.pyplot as plt
import dataset_utils as loader_data

train_dir = 'SVHN'
train_set = sio.loadmat(os.path.join(train_dir, 'train_32x32.mat'))

train_x = numpy.transpose(train_set['X'], [3, 0, 1, 2])
train_y = train_set['y']
train_y[train_y == 10] = 0
one_hot_y = loader_data.dense_to_one_hot(train_y, 10)

# plt.imshow(train_x[0])
# plt.show()
print(train_y[95:99])
print(one_hot_y[95:99])

