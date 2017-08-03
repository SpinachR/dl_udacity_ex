import os
import time
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import nn as model
import utils as utils
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('Agg')


class ConditionalGAN(object):
    def __init__(self, config):
        self.config = config
        self.mnist = input_data.read_data_sets('../MNIST', reshape=False, one_hot=True)
        self.build_model(config)

        if not os.path.exists(config.sampledir):
            os.makedirs(config.sampledir)

        if not os.path.exists(config.checkpoint_basename):
            os.makedirs(config.checkpoint_basename)

        if not os.path.exists(config.logdir):
            os.makedirs(config.logdir)

        self.loss_summaries = tf.summary.merge([
            tf.summary.scalar('d_loss_real', self.d_loss_real),
            tf.summary.scalar('d_loss_fake', self.d_loss_fake),
            tf.summary.scalar('d_loss', self.d_loss),
            tf.summary.scalar('g_loss', self.g_loss)
        ])

        self.histogram_summary = tf.summary.merge([
            tf.summary.histogram('z', self.z),
            tf.summary.histogram('d_real', self.D),
            tf.summary.histogram('d_fake', self.D_fake),
        ])

        self.image_summary = tf.summary.merge([
            tf.summary.image('G_image out of training', self.G)
        ])

        self.g_summary = tf.summary.merge([
            tf.summary.histogram('z', self.z),
            tf.summary.image('G_image in training', self.G),
            tf.summary.histogram('d_fake', self.D_fake),
            tf.summary.scalar('d_loss_fake', self.d_loss_fake),
            tf.summary.scalar('g_loss', self.g_loss)
        ])

        self.d_summary = tf.summary.merge([
            tf.summary.scalar('d_loss_real', self.d_loss_real),
            tf.summary.scalar('d_loss', self.d_loss),
            tf.summary.histogram('z', self.z),
            tf.summary.histogram('d_real', self.D),
        ])

        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.summary_writer = tf.summary.FileWriter(config.logdir, self.sess.graph)

    def build_model(self, config):
        image_dim = [config.x_height, config.x_width, config.x_depth]
        self.x = tf.placeholder(tf.float32, [config.batch_size]+image_dim, name='real_image')
        self.y = tf.placeholder(tf.float32, [None, config.y_dim], name='label')
        self.z = tf.placeholder(tf.float32, [None, config.z_dim], name='z')
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.G, _ = model.generator(self.z, self.y, is_training=self.is_training)
        self.D, self.D_logits, _ = model.discriminator(self.x, self.y, is_training=self.is_training)

        self.D_fake, self.D_fake_logits, _ = model.discriminator(self.G, self.y, reuse=True, is_training=self.is_training)

        with tf.variable_scope('d_loss'):
            self.d_loss_real = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.D),
                                                               logits=self.D_logits)
            self.d_loss_fake = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(self.D_fake),
                                                               logits=self.D_fake_logits)
            self.d_loss = self.d_loss_real + self.d_loss_fake

        with tf.variable_scope('g_loss'):
            self.g_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.D_fake),
                                                          logits=self.D_fake_logits)

        with tf.variable_scope('opt_D'):
            self.vars_D = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
            self.opt_D = tf.train.AdamOptimizer(config.lr_d, beta1=config.beta1).minimize(
                self.d_loss, var_list=self.vars_D
            )
        with tf.variable_scope('opt_G'):
            self.vars_G = [var for var in tf.trainable_variables() if 'generator' in var.name]
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

                while batch_num < data_array.num_examples//config.batch_size:
                    step = step+1
                    image, label = data_array.next_batch(config.batch_size)
                    z = utils.generate_z(config.batch_size, config.z_dim)

                    t1 = time.time()
                    # update D network
                    _, summary_d, cur_loss_d, cur_loss_d_real, cur_loss_d_fake = sess.run(
                        [self.opt_D, self.d_summary, self.d_loss, self.d_loss_real, self.d_loss_fake],
                        feed_dict={self.z: z, self.x: image, self.y: label, self.is_training: True})
                    self.summary_writer.add_summary(summary_d, step)

                    # update G network
                    _, summary_g, cur_loss_g = sess.run([self.opt_G, self.g_summary, self.g_loss],
                                                        feed_dict={
                                                            self.z: z,
                                                            self.y: label,
                                                            self.is_training: True
                                                        })
                    self.summary_writer.add_summary(summary_g, step)
                    t2 = time.time()

                    batch_num = batch_num + 1

                    if step % config.summary_every_n_steps == 0:
                        print("Finished {}/{}, Time:{:.2f}s, d_loss: {:.8f}, g_loss: {:.8f}".
                              format(step, config.max_steps, t2 - t1, cur_loss_d, cur_loss_g))

                    if step % config.sample_every_n_steps == 0:
                        z_ = utils.generate_z(64, config.z_dim)
                        label_ = utils.sample_label(64, 8)
                        samples, summary_image = sess.run([self.G, self.image_summary],
                                                          feed_dict={self.z: z_, self.y: label_, self.is_training: False})
                        self.summary_writer.add_summary(summary_image, step)
                        # utils.save_images(np.reshape(samples, newshape=[64, 28, 28]), [8, 8], config.sampledir + '/{}.png'.format(step))
                        fig = utils.plot(samples, 8, [28, 28])
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
