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
from tensorflow.examples.tutorials.mnist import input_data


# ______________1. Building Arguments Scope and blocks________________________
def _batch_norm_params(is_training):
    return {
        "decay": 0.9,
        "epsilon": 1e-5,
        "scale": True,
        "updates_collections": None,
        "is_training": is_training
    }


def _gen_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training),
                        outputs_collections=outputs_collections):
        with slim.arg_scope([slim.conv2d_transpose],
                            kernel_size=[4, 4], stride=2, padding="SAME") as arg_scp:
            return arg_scp


def _disc_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=utils.lrelu,
                        outputs_collections=outputs_collections):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=[4, 4], stride=2, padding='SAME') as arg_scp:
            return arg_scp


def generator_x(z_c, image_dim=784, is_training=True, scope='generator'):
    '''generator for x: use z to generate x
    Args:
        z_c: z(noise) + c(latent variables including continuous and categories codes)
        image_channel : output's dimension
    '''

    with tf.variable_scope(scope, default_name='generator') as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_gen_arg_scope(is_training, end_pts_collection)):
            # FC. 1024 RELU batchnorm
            h1 = slim.fully_connected(z_c, 128, scope='gx_h1_fc')

            h2 = slim.fully_connected(h1, 512, scope='gx_h2_fc')

            h3 = slim.fully_connected(h2, image_dim,
                                      activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='gx_h3_fc')

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return tf.nn.sigmoid(h3), h2, h1, end_pts


def generator_z(x, z_dim=64, is_training=True, scope='generator'):
    '''generator for z: inference network
    Args:
        x: image
        z_dim : latent variable's dimension
    '''

    with tf.variable_scope(scope, default_name='generator') as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_gen_arg_scope(is_training, end_pts_collection)):
            # FC 1024, lrelu, batchnorm
            h1 = slim.fully_connected(x, 512, scope='gz_h1_fc')

            h2 = slim.fully_connected(h1, 128, scope='gz_h2_fc')

            # Discriminator output: fc z_dim
            d_logit = slim.fully_connected(h2, z_dim, activation_fn=None, normalizer_fn=None,
                                           normalizer_params=None,
                                           scope='d_output_logit')

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return tf.nn.sigmoid(d_logit), h2, h1, end_pts


def discriminator(image, z, reuse=None, is_training=True, scope='discriminator'):
    '''discriminator
    Args:
        image: (None, ~).  mnist is (none, 784)
        z: (None, ~). latent variable
    :return
        D: logit of the joint distribution - p(image, z)
        output collections in specific steps
    '''
    with tf.variable_scope(scope, default_name='discriminator', reuse=reuse) as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_disc_arg_scope(is_training, end_pts_collection)):
            x = tf.concat([image, z], axis=1)
            print('image', image.get_shape())
            print('z', z.get_shape())
            print('x', x.get_shape())

            # FC 1024, lrelu, batchnorm
            h1 = slim.fully_connected(x, 128, scope='d_h1_fc')

            h2 = slim.fully_connected(h1, 512, scope='d_h2_fc')

            d_logit = slim.fully_connected(h1, 1, activation_fn=None, normalizer_fn=None,
                                           normalizer_params=None,
                                           scope='d_output_logit')

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return tf.nn.sigmoid(d_logit), d_logit, end_pts


# ______________2. Building Bidirectinal GAN Model________________________
class BiGAN(object):
    def __init__(self, config):
        self.config = config
        self.mnist = input_data.read_data_sets('../MNIST', reshape=True, one_hot=True)  # (none, 784)
        self.build_model(config)
        if not os.path.exists(config.sampledir):
            os.makedirs(config.sampledir)

        if not os.path.exists(config.checkpoint_basename):
            os.makedirs(config.checkpoint_basename)

        if not os.path.exists(config.logdir):
            os.makedirs(config.logdir)

        self.image_summary = tf.summary.merge([
            tf.summary.image('G_image out of training', tf.reshape(self.x_hat, [-1, 28, 28, 1]))
        ])

        self.g_summary = tf.summary.merge([
            tf.summary.histogram('z', self.z),
            tf.summary.image('G_image in training', tf.reshape(self.x_hat, [-1, 28, 28, 1])),
            tf.summary.histogram('d_fake', self.D_gen),
            tf.summary.scalar('d_loss_gen', self.d_loss_gen),
            tf.summary.scalar('g_loss', self.g_loss),
            tf.summary.scalar('g_loss_gen', self.g_loss_gen)
        ])

        self.d_summary = tf.summary.merge([
            tf.summary.scalar('d_loss_enc', self.d_loss_enc),
            tf.summary.scalar('g_loss_enc', self.g_loss_enc),
            tf.summary.scalar('d_loss', self.d_loss),
            tf.summary.histogram('z', self.z),
            tf.summary.histogram('d_real', self.D_enc),
        ])

        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.summary_writer = tf.summary.FileWriter(config.logdir, self.sess.graph)

    def build_model(self, config):
        image_dim = [config.x_height, config.x_width, config.x_depth]
        self.x = tf.placeholder(tf.float32, [None, config.x_dim], name='real_image')
        self.z = tf.placeholder(tf.float32, [None, config.z_dim], name='latent_var')
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # build graph
        self.x_hat, self.x_h2, self.x_h1, _ = generator_x(self.z, config.x_dim, self.is_training)
        self.z_hat, self.z_h2, self.z_h1, _ = generator_z(self.x, config.z_dim, self.is_training)

        print(self.x_hat.shape, self.x_h2.shape, self.x_h1.shape)
        print(self.z_hat.shape, self.z_h2.shape, self.z_h1.shape)

        mid_gen_x = tf.concat([self.x_h2, self.x_h1, self.z], axis=1)
        mid_gen_z = tf.concat([self.z_h1, self.z_h2, self.z_hat], axis=1)

        self.D_enc, self.D_enc_logit, _ = discriminator(self.x, self.z_hat, is_training=self.is_training)
        self.D_gen, self.D_gen_logit, _ = discriminator(self.x_hat, self.z, reuse=True, is_training=self.is_training)

        # define losses
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
            self.opt_D = tf.train.AdamOptimizer(config.lr_d, beta1=config.beta1).minimize(
                self.d_loss, var_list=self.vars_D
            )

        with tf.variable_scope('opt_G'):
            self.opt_G = tf.train.AdamOptimizer(config.lr_g, beta1=config.beta1).minimize(
                self.g_loss, var_list=self.vars_G
            )

    def fit(self):
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
            data_array = self.mnist.train  # containing image and label
            while e <= config.epoch:
                while batch_num < data_array.num_examples // config.batch_size:
                    step = step + 1
                    image, label = data_array.next_batch(config.batch_size)
                    #z = utils.generate_z(config.batch_size, config.z_dim)
                    z = np.random.uniform(-1, 1, size=[config.batch_size, config.z_dim])

                    # update D network
                    _, summary_d, cur_loss_d = sess.run(
                        [self.opt_D, self.d_summary, self.d_loss],
                        feed_dict={self.z: z, self.x: image, self.is_training: True})
                    self.summary_writer.add_summary(summary_d, step)

                    # update G network
                    _, summary_g, cur_loss_g = sess.run([self.opt_G, self.g_summary, self.g_loss],
                                                        feed_dict={
                                                            self.x: image,
                                                            self.z: z,
                                                            self.is_training: True
                                                        })
                    self.summary_writer.add_summary(summary_g, step)

                    batch_num = batch_num + 1
                    if step % config.summary_every_n_steps == 0:
                        print("Finished {},  d_loss: {:.8f}, g_loss: {:.8f}".
                              format(step, cur_loss_d, cur_loss_g))

                    if step % config.sample_every_n_steps == 0:
                        z_ = np.random.uniform(-1, 1, size=[81, config.z_dim])  # need to be identical with self.z
                        #z_ = utils.generate_z(81, config.z_dim)
                        # cat_ = utils.sample_label(81, 9)
                        # c_ = np.random.uniform(0, 1, size=[81, config.c_dim])
                        #c_ = utils.generate_c(81, config.c_dim)
                        samples, summary_image = sess.run([self.x_hat, self.image_summary],
                                                          feed_dict={self.z: z_, self.is_training: False})
                        self.summary_writer.add_summary(summary_image, step)
                        # utils.save_images(np.reshape(samples, newshape=[64, 28, 28]), [8, 8], config.sampledir + '/{}.png'.format(step))
                        fig = utils.plot(samples, 9, [28, 28])
                        plt.savefig(config.sampledir + '/{}.png'.format(step), bbox_inches='tight')
                        plt.close(fig)

                    if step % config.savemodel_every_n_steps == 0:
                        self.saver.save(sess, config.checkpoint_basename + '/' + config.model_ckpt,
                                        global_step=step)
                e = e+1
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