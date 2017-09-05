import matplotlib
matplotlib.use('Agg')
import dataset_utils as loader_data
import scipy
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.python.framework import dtypes


def plot1(samples, figsize, image_dim):
    fig = plt.figure(figsize=(figsize, figsize))
    gs = gridspec.GridSpec(figsize, figsize)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(image_dim))

    return fig

svhn = loader_data.load_svhn('SVHN', reshape=False, dtype=dtypes.uint8)

train_x, train_y = svhn.train.next_batch(81)
print(train_y)
fig = plot1(train_x, 9, [32, 32, 3])
plt.savefig('1.png', bbox_inches='tight')
plt.close(fig)



