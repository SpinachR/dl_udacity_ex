import matplotlib
matplotlib.use('Agg')
import os
import time
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import utils as utils
import dataset_utils as data_loader


# batchNorm layer parameters
def _batch_norm_params(is_training):
    return {
        "decay": 0.9,
        "epsilon": 1e-5,
        "scale": True,
        "updates_collections": None,
        "is_training": is_training
    }


# convolution layer scope
def _conv2d_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        activation_fn=utils.lrelu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training),
                        padding='VALID',
                        outputs_collections=outputs_collections) as arg_scp:
        return arg_scp


# convolution transpose layer scope
def _con2d_transpose_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d_transpose],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        activation_fn=utils.lrelu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training),
                        padding='VALID',
                        outputs_collections=outputs_collections) as arg_scp:
        return arg_scp


# Generate z given x
def G_z(x, is_training=True, scope='generator'):
    '''
    :param x: given image [32, 32, 3]
    :param is_training: training mode or not
    :param scope:
    :return: z [1, 1, 256]
    '''
    with tf.variable_scope(scope, default_name='generator') as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_conv2d_scope(is_training, end_pts_collection)):
            # [32, 32, 3]
            h1 = slim.conv2d(x, 32, kernel_size=5, stride=1, scope='conv2d_z_h1')

            # [28, 28, 32]
            h2 = slim.conv2d(h1, 64, kernel_size=4, stride=2, scope='conv2d_z_h2')

            # [13, 13, 64]
            h3 = slim.conv2d(h2, 128, kernel_size=4, stride=1, scope='conv2d_z_h3')

            # [10, 10, 128]
            h4 = slim.conv2d(h3, 256, kernel_size=4, stride=2, scope='conv2d_z_h4')

            # [4, 4, 256]
            h5 = slim.conv2d(h4, 512, kernel_size=4, stride=1, scope='conv2d_z_h5')

            # [1, 1, 512]
            h6 = slim.conv2d(h5, 512, kernel_size=1, stride=1, scope='conv2d_z_h6')

            # [1, 1, 512]
            h7 = slim.conv2d(h6, 256, kernel_size=1, stride=1, scope='conv2d_z_h7',
                             activation_fn=None, normalizer_fn=None, normalizer_params=None)

            return h7   # [1, 1, 256]


# Generate x given z
def G_x(z, is_training=True, scope='generator'):
    '''
    :param z: z [1, 1, 256] uniform distribution
    :param is_training: training mode or not
    :param scope:
    :return: x - fake image
    '''
    with tf.variable_scope(scope, default_name='generator') as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_con2d_transpose_scope(is_training, end_pts_collection)):
            # [1, 1, 256]
            h1 = slim.conv2d_transpose(z, 256, kernel_size=4, stride=1)

            # [4, 4, 256]
            h2 = slim.conv2d_transpose(h1, 128, kernel_size=4, stride=2)

            # [10, 10, 128]
            h3 = slim.conv2d_transpose(h2, 64, kernel_size=4, stride=1)

            # [13, 13, 64]
            h4 = slim.conv2d_transpose(h3, 32, kernel_size=4, stride=2)

            # [28, 28, 32]
            h5 = slim.conv2d_transpose(h4, 32, kernel_size=5, stride=1)

            # [32, 32, 32]
            h6 = slim.conv2d(h5, 32, kernel_size=1, stride=1, padding='VALID',
                             weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             activation_fn=utils.lrelu,
                             normalizer_fn=slim.batch_norm,
                             normalizer_params=_batch_norm_params(is_training))

            # [32, 32, 32]
            h7 = slim.conv2d(h6, 3, kernel_size=1, stride=1, padding='VALID',
                             weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             activation_fn=tf.nn.sigmoid)
            return h7  # [32, 32, 3]


# Discriminator for (x, z) pair
def D(x, z, keep_prob, reuse=None, is_training=True, scope='discriminator'):
    with tf.variable_scope(scope, default_name='discriminator', reuse=reuse) as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_conv2d_scope(is_training, end_pts_collection)):
            ## D(x) - [32, 32, 3]
            x_h1 = slim.conv2d(x, 32, kernel_size=5, stride=1, normalizer_fn=None, normalizer_params=None, scope='dx_h1')
            x_h1 = tf.nn.dropout(x_h1, keep_prob)

            # [28, 28, 32]
            x_h2 = slim.conv2d(x_h1, 64, kernel_size=4, stride=2, scope='dx_h2')
            x_h2 = tf.nn.dropout(x_h2, keep_prob)

            # [13, 13, 64]
            x_h3 = slim.conv2d(x_h2, 128, kernel_size=4, stride=1, scope='dx_h3')
            x_h3 = tf.nn.dropout(x_h3, keep_prob)

            # [10, 10, 128]
            x_h4 = slim.conv2d(x_h3, 256, kernel_size=4, stride=2, scope='dx_h4')
            x_h4 = tf.nn.dropout(x_h4, keep_prob)

            # [4, 4, 256]
            x_h5 = slim.conv2d(x_h4, 512, kernel_size=4, stride=1, scope='dx_h5')
            x_h5 = tf.nn.dropout(x_h5, keep_prob)   # [1, 1, 512]

            ## D(z) - [1, 1. 256]
            z_h1 = slim.conv2d(z, 512, kernel_size=1, stride=1, normalizer_fn=None, normalizer_params=None, scope='dz_h1')
            z_h1 = tf.nn.dropout(z_h1, keep_prob)

            # [1, 1, 512]
            z_h2 = slim.conv2d(z_h1, 512, kernel_size=1, stride=1, scope='dz_h2', normalizer_fn=None, normalizer_params=None)
            z_h2 = tf.nn.dropout(z_h2, keep_prob)

            ## D(x, z) - concatenate [D(x), D(z)] (1, 1, 1024)
            inputxz = tf.concat([x_h5, z_h2], axis=2)
            xz_h1 = slim.conv2d(inputxz, 1024, kernel_size=1, stride=1, scope='dxz_h1', normalizer_fn=None, normalizer_params=None)
            xz_h1 = tf.nn.dropout(xz_h1, keep_prob)

            xz_h2 = slim.conv2d(xz_h1, 1024, kernel_size=1, stride=1, scope='dxz_h2', normalizer_fn=None, normalizer_params=None)
            xz_h2 = tf.nn.dropout(xz_h2, keep_prob)

            xz_output = slim.conv2d(xz_h2, 1, kernel_size=1, stride=1, scope='dxz_output', activation_fn=None,
                                    normalizer_fn=None, normalizer_params=None)
            xz_output = tf.nn.dropout(xz_output, keep_prob)

            return tf.nn.sigmoid(xz_output), xz_output


# ______________2. Building Bidirectinal GAN Model________________________
class aliGAN(object):
    def __init__(self, config):
        self.config = config
        self.svhn = data_loader.load_svhn('../SVHN')
        self.build_model(config)
        if not os.path.exists(config.sampledir):
            os.makedirs(config.sampledir)

        if not os.path.exists(config.checkpoint_basename):
            os.makedirs(config.checkpoint_basename)

        if not os.path.exists(config.logdir):
            os.makedirs(config.logdir)

        self.image_summary = tf.summary.merge([
            tf.summary.image('G_image out of training', self.x_hat)
        ])

        self.g_summary = tf.summary.merge([
            tf.summary.histogram('z_G', self.z),
            tf.summary.image('G_image in training', self.x_hat),
            tf.summary.histogram('d_fake', self.D_gen),
            tf.summary.scalar('g_loss', self.g_loss)
        ])

        self.d_summary = tf.summary.merge([
            tf.summary.scalar('d_loss', self.d_loss),
            tf.summary.histogram('z_D', self.z),
            tf.summary.histogram('d_real', self.D_enc),
        ])

        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.summary_writer = tf.summary.FileWriter(config.logdir, self.sess.graph)

    def build_model(self, config):
        image_dim = [config.x_height, config.x_width, config.x_depth]
        z_dim = [1, 1, config.z_dim]

        self.x = tf.placeholder(tf.float32, [None, config.x_height, config.x_width, config.x_depth], name='real_image')
        self.z = tf.placeholder(tf.float32, [None, 1, 1, config.z_dim], name='latent_var')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # build graph
        self.z_hat = G_z(self.x, self.is_training)
        self.x_hat = G_x(self.z, self.is_training)

        self.D_enc, self.D_enc_logit = D(self.x, self.z_hat, self.keep_prob, is_training=self.is_training)
        self.D_gen, self.D_gen_logit = D(self.x_hat, self.z, self.keep_prob, reuse=True, is_training=self.is_training)

        # define loss
        with tf.variable_scope('d_loss'):
            self.d_loss_enc = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.D_enc),
                                                              logits=self.D_enc_logit)
            self.d_loss_gen = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(self.D_gen),
                                                              logits=self.D_gen_logit)

            self.d_loss = self.d_loss_enc + self.d_loss_gen

        with tf.variable_scope('g_loss'):
            self.g_loss_gen = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.D_gen),
                                                              logits=self.D_gen_logit)
            self.g_loss_enc = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(self.D_enc),
                                                              logits=self.D_enc_logit)
            self.g_loss = self.g_loss_gen + self.g_loss_enc

        # define optimizer
        self.vars_D = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        self.vars_G = [var for var in tf.trainable_variables() if 'generator' in var.name]
        with tf.variable_scope('opt_D'):
            self.opt_D = tf.train.AdamOptimizer(config.lr_d, beta1=config.beta1, beta2=config.beta2).minimize(
                self.d_loss, var_list=self.vars_D
            )

        with tf.variable_scope('opt_G'):
            self.opt_G = tf.train.AdamOptimizer(config.lr_g, beta1=config.beta1, beta2=config.beta2).minimize(
                self.g_loss, var_list=self.vars_G
            )

    def train(self):
        config = self.config
        with self.sess as sess:
            if config.load_ckpt is True:
                could_load, path = self.load_latest_checkpoint(config.checkpoint_basename+'/'+config.model_ckpt)
                if could_load:
                    print(" [*] Load SUCCESS", path)
                else:
                    print(" [!] Load failed...")
            else:
                sess.run(tf.global_variables_initializer())

            e = 0
            step = 0
            batch_num = 0
            data_array = self.svhn.train  # containing image and label
            while e <= config.epoch:
                while batch_num < data_array.num_examples // config.batch_size + 1:
                    step = step + 1
                    image, label = data_array.next_batch(config.batch_size)
                    z = np.random.uniform(-1, 1, size=[config.batch_size, 1, 1, config.z_dim])

                    # update D network
                    _, summary_d, cur_loss_d = sess.run(
                        [self.opt_D, self.d_summary, self.d_loss],
                        feed_dict={self.z: z, self.x: image, self.is_training: True, self.keep_prob: 0.8})
                    self.summary_writer.add_summary(summary_d, step)

                    # update G network
                    _, summary_g, cur_loss_g = sess.run([self.opt_G, self.g_summary, self.g_loss],
                                                        feed_dict={
                                                            self.x: image,
                                                            self.z: z,
                                                            self.is_training: True,
                                                            self.keep_prob: 0.8
                                                        })
                    self.summary_writer.add_summary(summary_g, step)

                    batch_num = batch_num + 1
                    if step % config.summary_every_n_steps == 0:
                        print("Finished {},  d_loss: {:.8f}, g_loss: {:.8f}".
                              format(step, cur_loss_d, cur_loss_g))

                    if step % config.sample_every_n_steps == 0:
                        z_ = np.random.uniform(-1, 1, size=[81, 1, 1, config.z_dim])  # need to be identical with self.z
                        # cat_ = utils.sample_label(81, 9)
                        # c_ = np.random.uniform(0, 1, size=[81, config.c_dim])
                        # c_ = utils.generate_c(81, config.c_dim)
                        samples, summary_image = sess.run([self.x_hat, self.image_summary],
                                                          feed_dict={self.z: z_, self.is_training: False})
                        self.summary_writer.add_summary(summary_image, step)
                        # utils.save_images(np.reshape(samples, newshape=[64, 28, 28]), [8, 8], config.sampledir + '/{}.png'.format(step))
                        fig = utils.plot(samples, 9, [32, 32, 3])
                        plt.savefig(config.sampledir + '/{}.png'.format(step), bbox_inches='tight')
                        plt.close(fig)

                    if step % config.savemodel_every_n_steps == 0:
                        self.saver.save(sess, config.checkpoint_basename + '/' + config.model_ckpt,
                                        global_step=step)
                e = e + 1
                batch_num = 0

    def load_latest_checkpoint(self, ckpt_dir, exclude=None):
        path = tf.train.latest_checkpoint(ckpt_dir)
        if path is None:
            raise AssertionError("No ckpt exists in {0}.".format(ckpt_dir))
        self._load(path, exclude)
        return True, path

    def load_from_path(self, ckpt_path, exclude=None):
        self._load(ckpt_path, exclude)

    def _load(self, ckpt_path, exclude):
        init_fn = slim.assign_from_checkpoint_fn(ckpt_path,
                                                 slim.get_variables_to_restore(exclude=exclude),
                                                 ignore_missing_vars=True)
        init_fn(self.sess)

