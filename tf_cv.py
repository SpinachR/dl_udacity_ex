import tensorflow as tf
import numpy as np
import scipy.misc
import math


def augment_data(data, params):
    if params['no_aug']:
        return data

    if params['random_brightness']:
        print('apply random_brightness...', flush=True)
        data = tf.map_fn(lambda img: tf.image.random_brightness(img, params['random_brightness']), data)

    if params['random_saturation']:
        print('apply random_saturation...', flush=True)
        data = tf.map_fn(lambda img: tf.image.random_saturation(img, lower=params['random_saturation'][0],
                                                                upper=params['random_saturation'][1]), data)

    if params['random_hue']:
        print('apply random_hue...', flush=True)
        data = tf.map_fn(lambda img: tf.image.random_hue(img, params['random_hue']), data)

    if params['random_contrast']:
        print('apply random_contrast...', flush=True)
        data = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=params['random_contrast'][0],
                                                              upper=params['random_contrast'][1]), data)

    if params['intens_flip']:
        print('apply intensity flip...', flush=True)
        col_factor = tf.random_uniform(shape=(tf.shape(data)[0], 1, 1, 1), minval=0, maxval=2, dtype=tf.int32) * 2 - 1
        data = data * tf.cast(col_factor, tf.float32)

    if params['intens_scale']:
        print('apply intens_scale ...', flush=True)
        col_factor = tf.random_uniform(shape=(tf.shape(data)[0], 1, 1, 1), minval=params['intens_scale'][0],
                                       maxval=params['intens_scale'][1])
        data = data * col_factor

    if params['intens_offset']:
        print('apply intens_offset ...', flush=True)
        col_offset = tf.random_uniform(shape=(tf.shape(data)[0], 1, 1, 1), minval=params['intens_offset'][0],
                                       maxval=params['intens_offset'][1])
        data = data + col_offset

    if params['gaussian_noise_std'] > 0.0:
        print('apply gaussian_noise ...', flush=True)
        data = data + tf.random_normal(shape=tf.shape(data), mean=0.0, stddev=params['gaussian_noise_std'],
                                       dtype=tf.float32)

    return data  # data = tf.clip_by_value(data, clip_value_min=0.0, clip_value_max=1.0)


def tf_kernel_prep_3d(kernel, n_channels):
    return np.tile(kernel, (n_channels, 1, 1)).swapaxes(0, 1).swapaxes(1, 2)


def tf_deriv(batch, ksize=3, padding='SAME'):
    n_ch = int(batch.get_shape().as_list()[3])
    gx = tf_kernel_prep_3d(np.array([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]), n_ch)
    gy = tf_kernel_prep_3d(np.array([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]]), n_ch)
    kernel = tf.constant(np.stack([gx, gy], axis=-1), name="DerivKernel_image", dtype=np.float32)
    return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding, name="GradXY")


def merge_one(images, columns, rows):
    n, h, w, c = images.shape
    # grid_size = int(math.sqrt(n))
    num = columns * rows
    images = images[:num]

    img = np.zeros((h * rows, w * columns, c))

    for idx, image in enumerate(images):
        i = idx % columns
        j = idx // columns
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def merge_two(images_1, images_2):
    n, h, w, c = images_1.shape
    rows = int(math.sqrt(n))
    img = np.zeros([rows * h, rows * 2 * w, c])

    imgs_1 = images_1[:rows ** 2]
    imgs_2 = images_2[:rows ** 2]

    for idx, (s, t) in enumerate(zip(imgs_1, imgs_2)):
        i = idx // rows
        j = idx % rows
        img[i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h, :] = s
        img[i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h, :] = t

    return img


def save_images(images, path1):
    scipy.misc.imsave(path1, images)
