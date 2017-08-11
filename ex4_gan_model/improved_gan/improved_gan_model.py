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
import weight_norm_nn as nn


def log_sum_exp(x, axis=1):
    '''
    :return: log[sum_i(exp(x_i))]
    '''
    m = tf.reduce_max(x, axis=axis)
    return m + tf.log(tf.reduce_sum(tf.exp(x - tf.expand_dims(m, 1)), axis=axis))


# ______________1. Building Arguments Scope and blocks________________________
def _batch_norm_params(is_training):
    return {
        "decay": 0.9,
        "epsilon": 1e-5,
        "scale": True,
        "updates_collections": None,
        "is_training": is_training
    }


def generator_tim(z_c, is_training=True, scope='generator'):
    with tf.variable_scope(scope, default_name='generator') as scp:
        # z = tf.reshape(z, shape=[-1, 1, 1, z_dim])  # which is different from doing project first
        h1 = slim.fully_connected(z_c, 500, activation_fn=tf.nn.softplus,
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params=_batch_norm_params(is_training), scope='g_h1_fc')

        h2 = slim.fully_connected(h1, 500, activation_fn=tf.nn.softplus,
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params=_batch_norm_params(is_training), scope='g_h2_fc')

        h3 = slim.fully_connected(h2, 28**2, activation_fn=None, scope='g_h3_fc')

        return tf.nn.sigmoid(h3), h3


def discriminator_tim(image, n=10, reuse=None, is_training=True, scope='discriminator'):
    with tf.variable_scope(scope, default_name='discriminator', reuse=reuse):
        with slim.arg_scope([nn.gaussian_noise_layer], is_training=is_training, sigma=0.5):
            h1 = nn.gaussian_noise_layer(image, sigma=0.3)
            print('h1: ', h1.get_shape())
            print('h1 name: ', h1.name)
            h1 = slim.fully_connected(h1, 1000, scope='d_h1_fc')

            h2 = nn.gaussian_noise_layer(h1)
            h2 = slim.fully_connected(h2, 500, scope='d_h2_fc')

            h3 = nn.gaussian_noise_layer(h2)
            h3 = slim.fully_connected(h3, 250, scope='d_h3_fc')

            h4 = nn.gaussian_noise_layer(h3)
            h4 = slim.fully_connected(h4, 250, scope='d_h4_fc')

            h5 = nn.gaussian_noise_layer(h4)
            h5 = slim.fully_connected(h5, 250, scope='d_h5_fc')

            h6 = nn.gaussian_noise_layer(h5)
            h6 = slim.fully_connected(h6, n, activation_fn=None, scope='d_h6_fc')

            return tf.nn.softmax(h6), h6, h4

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''


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
                        outputs_collections=outputs_collections,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training)):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=[4, 4], stride=2, padding='SAME') as arg_scp:
            return arg_scp


def generator(z_c, image_channel=1, is_training=True, scope='generator'):
    '''generator
    Args:
        z_c: random noise to generate images
        image_channel : output's dimension
    '''

    with tf.variable_scope(scope, default_name='generator') as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_gen_arg_scope(is_training, end_pts_collection)):
            # z = tf.reshape(z, shape=[-1, 1, 1, z_dim])  # which is different from doing project first
            # FC. 1024 RELU batchnorm
            h1 = slim.fully_connected(z_c, 1024, scope='g_h1_fc')

            # FC. 7*7*128 RELU. batchnorm
            h2 = slim.fully_connected(h1, 7 * 7 * 128, scope='g_h2_fc')

            # reshape
            h2 = tf.reshape(h2, [-1, 7, 7, 128])

            # 4*4 upconv. 64 RELU, stride 2, batchnorm
            h3 = slim.conv2d_transpose(h2, 64, scope='g_h2_con2d_transpose')

            # 4*4 upconv. 1 channel
            h4 = slim.conv2d_transpose(h3, image_channel, scope='g_h4_con2d_transpose',
                                       activation_fn=None, normalizer_fn=None, normalizer_params=None)
            # now [batch_size, 28, 28, 1]

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return tf.nn.sigmoid(h4), end_pts


def discriminator_and_q(image, n=10, reuse=None, is_training=True, scope='discriminator'):
    '''discriminator
    Args:
        image: (None, ~, ~, ~).  mnist is (none, 28, 28, 1)
        n: num of classes
    :return
        D: {d_probability, d_logit}
        h2: middle features
        output collections in specific steps
    '''
    with tf.variable_scope(scope, default_name='discriminator', reuse=reuse) as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_disc_arg_scope(is_training, end_pts_collection)):
            # 4*4 conv, 64, lrelu, stride 2
            h1 = slim.conv2d(image, 64, normalizer_fn=None, normalizer_params=None, scope='d_h1_conv')

            # 4*4 conv, 128, lrelu, stride 2, batchnorm
            h2 = slim.conv2d(h1, 128, scope='d_h2_conv')

            # flatten
            h2 = slim.flatten(h2, scope='d_h2_flatten')

            # FC 1024, lrelu, batchnorm
            h3 = slim.fully_connected(h2, 1024, scope='d_h3_fc')

            # Discriminator output: fc n
            d_logit = slim.fully_connected(h3, n, activation_fn=None, normalizer_fn=None,
                                           normalizer_params=None,
                                           scope='d_output_logit')

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return tf.nn.softmax(d_logit), d_logit, h2, end_pts


# ______________2. Building improved GAN Model________________________
class ImprovedGAN(object):
    def __init__(self, config):
        self.config = config
        self.mnist = input_data.read_data_sets('../MNIST', reshape=True, one_hot=False)  # x: 1-D, y: 1-D
        self.data_rng = np.random.RandomState(config.seed_data)  # for data
        self.rng = np.random.RandomState(config.seed)
        self.build_model(config)

        if not os.path.exists(config.sampledir):
            os.makedirs(config.sampledir)

        if not os.path.exists(config.checkpoint_basename):
            os.makedirs(config.checkpoint_basename)

        if not os.path.exists(config.logdir):
            os.makedirs(config.logdir)

        self.image_summary = tf.summary.merge([
            tf.summary.image('G_image out of training', tf.reshape(self.fake_image, [-1, 28, 28, 1]))
        ])

        self.g_summary = tf.summary.merge([
            tf.summary.histogram('z_g', self.z),
            tf.summary.image('G_image in training', tf.reshape(self.fake_image, [-1, 28, 28, 1])),
            tf.summary.histogram('D_logit_fake', self.D_logit_fake),
            tf.summary.scalar('g_loss', self.g_loss)
        ])

        self.d_summary = tf.summary.merge([
            tf.summary.histogram('D_logit_lab', self.D_logit_lab),
            tf.summary.histogram('D_logit_unl', self.D_logit_unl),
            tf.summary.scalar('loss_unl', self.loss_unl),
            tf.summary.scalar('loss_lab', self.loss_lab),
            tf.summary.scalar('d_loss', self.d_loss),
            tf.summary.histogram('z_d', self.z),
            tf.summary.scalar('error', self.train_err)
        ])

        self.test_summary = tf.summary.merge([
            tf.summary.scalar('test error', self.train_err)
        ])

        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.summary_writer = tf.summary.FileWriter(config.logdir, self.sess.graph)

    def build_model(self, config):
        labeled_batch = config.num_of_classes * config.num_of_labeled_data_for_each_classes
        # placeholder
        image_dim = [config.x_height, config.x_width, config.x_depth]
        self.x_lab = tf.placeholder(tf.float32, [None, config.x_dim], name='x_lab')
        self.lab = tf.placeholder(tf.int32, shape=[None], name='lab')  # not one-hot vector

        self.x_unl = tf.placeholder(tf.float32, [config.unlabeled_batch, config.x_dim], name='x_unl')

        self.z = tf.placeholder(tf.float32, [None, config.z_dim], name='input_G')
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # build graph
        self.fake_image, _ = generator_tim(self.z, is_training=self.is_training)
        self.D_prob_lab, self.D_logit_lab, _ = discriminator_tim(
            self.x_lab, config.num_of_classes, is_training=self.is_training)
        self.D_prob_unl, self.D_logit_unl, self.mid_real_features_unl = discriminator_tim(
            self.x_unl, config.num_of_classes, reuse=True, is_training=self.is_training)

        self.D_prob_fake, self.D_logit_fake, self.mid_fake_features_fake = discriminator_tim(
            self.fake_image, config.num_of_classes, reuse=True, is_training=self.is_training)

        # define losses
        z_exp_lab = tf.reduce_mean(log_sum_exp(self.D_logit_lab))  # mean[log_sum_exp(x_n)]
        z_exp_unl = tf.reduce_mean(log_sum_exp(self.D_logit_unl))
        z_exp_fake = tf.reduce_mean(log_sum_exp(self.D_logit_fake))

        with tf.variable_scope('d_loss'):
            indices = tf.stack([tf.range(labeled_batch), self.lab], 1)
            l_lab = tf.gather_nd(self.D_logit_lab, indices)  # find logit of corresponding label
            # l_lab = tf.gather(self.D_logit_lab, indices=[tf.range(labeled_batch), self.lab])
            print(l_lab.get_shape())
            l_unl = log_sum_exp(self.D_logit_unl)

            # tricks of computing softmax: sum(logit) - sum(log_sum_exp(logit))
            self.loss_lab = -tf.reduce_mean(l_lab) + tf.reduce_mean(z_exp_lab)
            # softplu:  ln(1+exp(x))
            self.loss_unl = -0.5 * tf.reduce_mean(l_unl) + \
                            0.5 * tf.reduce_mean(tf.nn.softplus(log_sum_exp(self.D_logit_unl))) + \
                            0.5 * tf.reduce_mean(tf.nn.softplus(log_sum_exp(self.D_logit_fake)))
            self.d_loss = self.loss_lab + self.loss_unl

            predict_label = tf.to_int32(tf.argmax(self.D_logit_lab, 1))
            self.train_err = tf.reduce_mean(tf.cast(tf.not_equal(predict_label, self.lab), tf.float32))

        with tf.variable_scope('g_loss'):
            mid_fake = tf.reduce_mean(self.mid_fake_features_fake, axis=0)
            mid_real_unl = tf.reduce_mean(self.mid_real_features_unl, axis=0)
            self.g_loss = tf.reduce_mean(tf.square(mid_real_unl - mid_fake))

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

    def train(self):
        config = self.config
        trainset = self.mnist.train
        testset = self.mnist.test
        # select labeled data

        inds = self.data_rng.permutation(trainset.num_examples)
        trainx = trainset.images[inds]
        trainy = trainset.labels[inds]
        txs = []
        tys = []
        for j in range(10):
            txs.append(trainx[trainy == j][:config.num_of_labeled_data_for_each_classes])
            tys.append(trainy[trainy == j][:config.num_of_labeled_data_for_each_classes])
        # obtain labeled data
        txs = np.concatenate(txs, axis=0)
        tys = np.concatenate(tys, axis=0)
        print('txs: ', txs.shape)
        print('tys: ', tys.shape)

        nr_batches_train = int(trainset.num_examples / config.unlabeled_batch)

        with self.sess as sess:
            if config.load_ckpt is True:
                could_load, path = self.load_latest_checkpoint(config.checkpoint_basename + '/' + config.model_ckpt)
                if could_load:
                    print(" [*] Load SUCCESS", path)
                else:
                    print(" [!] Load failed...")
            else:
                sess.run(tf.global_variables_initializer())

            e = 0
            step = 0
            batch_num = 0
            while e <= config.epoch:

                for t in range(nr_batches_train):
                    step = step + 1
                    # construct labeled data
                    inds = self.rng.permutation(txs.shape[0])
                    x_lab = txs[inds]
                    lab = tys[inds]

                    # construct unlabeled data; eliminate the labels here
                    x_unl, _ = trainset.next_batch(config.unlabeled_batch)
                    z = utils.generate_z(config.unlabeled_batch, config.z_dim)

                    t1 = time.time()
                    # update D network
                    _, cur_d_loss, cur_d_summary, cur_loss_lab, cur_loss_unl, cur_train_err = sess.run(
                        [self.opt_D, self.d_loss, self.d_summary, self.loss_lab, self.loss_unl, self.train_err],
                        feed_dict={self.z: z,
                                   self.x_lab: x_lab,
                                   self.lab: lab,
                                   self.x_unl: x_unl,
                                   self.is_training: True})
                    self.summary_writer.add_summary(cur_d_summary, step)

                    # update G network
                    _, cur_g_loss, cur_g_summary = sess.run([self.opt_G, self.g_loss, self.g_summary],
                                                            feed_dict={self.z: z,
                                                                       self.x_unl: x_unl,
                                                                       self.is_training: True})
                    self.summary_writer.add_summary(cur_g_summary, step)

                    t2 = time.time()
                    if step % config.summary_every_n_steps == 0:
                        test_err, test_summary = sess.run([self.train_err, self.test_summary],
                                                          feed_dict={self.x_lab: testset.images,
                                                                     self.lab: testset.labels,
                                                                     self.is_training: False})
                        self.summary_writer.add_summary(test_summary, step)
                        print("Finished {}, Time:{:.2f}s, d_loss: {:.8f}, g_loss: {:.8f}, "
                              "| lab_loss: {:.8f} and unl_loss: {:.8f} | train_err: {:.2f} | test_error: {:.2f}".
                              format(step, t2 - t1, cur_d_loss, cur_g_loss, cur_loss_lab, cur_loss_unl, cur_train_err, test_err))

                    if step % config.sample_every_n_steps == 0:
                        z_ = utils.generate_z(81, config.z_dim)
                        samples, summary_image = sess.run([self.fake_image, self.image_summary],
                                                          feed_dict={self.z: z_, self.is_training: False})
                        self.summary_writer.add_summary(summary_image, step)
                        # utils.save_images(np.reshape(samples, newshape=[64, 28, 28]), [8, 8], config.sampledir + '/{}.png'.format(step))
                        fig = utils.plot(samples, 9, [28, 28])
                        plt.savefig(config.sampledir + '/{}.png'.format(step), bbox_inches='tight')
                        plt.close(fig)

                    if step % config.savemodel_every_n_steps == 0:
                        self.saver.save(sess, config.checkpoint_basename + '/' + config.model_ckpt,
                                        global_step=step)

                e = e + 1

    def test(self):
        config = self.config
        testset = self.mnist.test
        with self.sess as sess:
            print(config.checkpoint_basename + '/' + config.model_ckpt)
            could_load, path = self.load_latest_checkpoint(config.checkpoint_basename)
            if could_load:
                print(" [*] Load SUCCESS", path)
            else:
                print(" [!] Load failed...")
                return
            error = []
            test_image, test_lab = testset.next_batch(100)
            test_err = sess.run([self.train_err], feed_dict={self.x_lab: testset.images,
                                                             self.lab: testset.labels,
                                                             self.is_training: False})
            print('test_error is ', test_err)

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
