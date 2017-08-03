import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils as utils


def _batch_norm_params(is_training):
    return {
        "decay": 0.9,
        "epsilon": 1e-5,
        "scale": True,
        "updates_collections": None,
        "is_training": is_training
    }


def _disc_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=utils.lrelu,
                        outputs_collections=outputs_collections,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training)):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=[5, 5], stride=2, padding='SAME') as arg_scp:
            return arg_scp


def discriminator(image, y, y_dim=10, df_dim=10, reuse=None, is_training=True, scope='discriminator'):
    '''discriminator
    Args:
        image: (None, ~, ~, ~).  mnist is (none, 28, 28, 1)
        y: label or attribute | y_dim: dimension of y
        df_dim: discriminator's first conv layer's output dim
    '''
    with tf.variable_scope(scope, default_name='discriminator', reuse=reuse) as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_disc_arg_scope(is_training, end_pts_collection)):
            yb = tf.reshape(y, shape=[-1, 1, 1, y_dim])
            # concat
            data_array = utils.conv_cond_concat(image, yb)

            h1 = slim.conv2d(data_array, df_dim, scope='d_h1_conv')
            h1 = utils.conv_cond_concat(h1, yb)

            h2 = slim.conv2d(h1, df_dim*2, scope='d_h2_conv')
            h2 = slim.flatten(h2, scope='d_h2_flatten')
            h2 = tf.concat([h2, y], axis=1)

            h3 = slim.fully_connected(h2, 1024, scope='d_h3_fc')
            h3 = tf.concat([h3, y], axis=1)

            h4 = slim.fully_connected(h3, 1, activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='d_h4_fc_lin')
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return tf.nn.sigmoid(h4), h4, end_pts


def _gen_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training),
                        outputs_collections=outputs_collections):
        with slim.arg_scope([slim.conv2d_transpose],
                            kernel_size=[5, 5], stride=2, padding="SAME") as arg_scp:
            return arg_scp


def generator(z, y, y_dim=10, is_training=True, scope='generator'):
    '''generator
    Args:
        z: noise input (None, z_dim)
        y: label or attribute | y_dim: dimension of y
    '''

    with tf.variable_scope(scope, default_name='generator') as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_gen_arg_scope(is_training, end_pts_collection)):
            z = tf.concat([z, y], 1)
            yb = tf.reshape(y, shape=[-1, 1, 1, y_dim])

            h1 = slim.fully_connected(z, 1024, scope='g_h1_fc')
            h1 = tf.concat([h1, y], axis=1)

            h2 = slim.fully_connected(h1, 7*7*2*64, scope='g_h2_fc')
            h2 = tf.reshape(h2, [-1, 7, 7, 64*2])
            h2 = utils.conv_cond_concat(h2, yb)

            h3 = slim.conv2d_transpose(h2, 128, scope='g_h3_con2d_transpose')
            # now [batch_size, 14, 14, 128]
            h3 = utils.conv_cond_concat(h3, yb)  # now [batch_size, 14, 14, 138]

            h4 = slim.conv2d_transpose(h3, 1, scope='g_h4_con2d_transpose',
                                       activation_fn=None, normalizer_fn=None, normalizer_params=None)
            # now [batch_size, 28, 28, 1]

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return tf.nn.sigmoid(h4), end_pts

