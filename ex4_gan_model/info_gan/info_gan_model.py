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
                        outputs_collections=outputs_collections,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training)):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=[4, 4], stride=2, padding='SAME') as arg_scp:
            return arg_scp


def generator(z_c, image_channel=1, is_training=True, scope='generator'):
    '''generator
    Args:
        z_c: z(noise) + c(latent variables including continuous and categories codes)
        image_channel : output's dimension
    '''

    with tf.variable_scope(scope, default_name='generator') as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_gen_arg_scope(is_training, end_pts_collection)):
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


def discriminator_and_q(image, cat_dim=10, c_dim=2, reuse=None, is_training=True, scope='discriminator'):
    '''discriminator
    Args:
        image: (None, ~, ~, ~).  mnist is (none, 28, 28, 1)
        cat_dim: dimensions of category variables
        c_dim: dimensions of continuous variables

    :return
        D: {d_out, d_logit}
        Q: {category_prob, continuous_var}
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

            # Discriminator output: fc 1
            d_logit = slim.fully_connected(h3, 1, activation_fn=None, normalizer_fn=None,
                                           normalizer_params=None,
                                           scope='d_output_logit')

            # output for Recognition network Q: fc, 128-batchnorm-lrelu
            q_a = slim.fully_connected(h3, 128, scope='q_fc_recognition')

            # compute probabilities for each category
            q_cat_logit = slim.fully_connected(q_a, cat_dim,
                                               activation_fn=None,
                                               normalizer_fn=None,
                                               normalizer_params=None,
                                               scope='q_cat_logit_recognition')

            # compute continuous variables
            q_c = slim.fully_connected(q_a, c_dim, activation_fn=tf.nn.sigmoid,
                                       normalizer_fn=None, normalizer_params=None, scope='q_c_recognition')

            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return tf.nn.sigmoid(d_logit), d_logit, tf.nn.softmax(q_cat_logit), q_cat_logit, q_c, end_pts


# ______________2. Building Info GAN Model________________________
class InfoGAN(object):
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

        self.image_summary = tf.summary.merge([
            tf.summary.image('G_image out of training', self.G_image)
        ])

        self.g_summary = tf.summary.merge([
            tf.summary.histogram('z', self.z),
            tf.summary.histogram('cat', self.cat),
            tf.summary.histogram('c', self.c),
            tf.summary.image('G_image in training', self.G_image),
            tf.summary.histogram('d_fake', self.D_fake),
            tf.summary.scalar('d_loss_fake', self.d_loss_fake),
            tf.summary.scalar('g_loss', self.g_loss)
        ])

        self.d_summary = tf.summary.merge([
            tf.summary.scalar('d_loss_real', self.d_loss_real),
            tf.summary.scalar('d_loss', self.d_loss),
            tf.summary.histogram('z', self.z),
            tf.summary.histogram('cat', self.cat),
            tf.summary.histogram('c', self.c),
            tf.summary.histogram('d_real', self.D),
        ])

        self.q_summary = tf.summary.merge([
            tf.summary.histogram('Qg_cat_logit', self.Qg_cat_logit),
            tf.summary.histogram('Qg_c', self.Qg_c),
            #tf.summary.scalar('cat_loss', self.cat_loss),
            #tf.summary.scalar('c_loss', self.c_loss),
            tf.summary.scalar('q_loss', self.q_loss),
            tf.summary.histogram('z', self.z),
            tf.summary.histogram('cat', self.cat),
            tf.summary.histogram('c', self.c),
        ])

        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.summary_writer = tf.summary.FileWriter(config.logdir, self.sess.graph)

    def build_model(self, config):
        # placeholder
        image_dim = [config.x_height, config.x_width, config.x_depth]
        self.x = tf.placeholder(tf.float32, [config.batch_size] + image_dim, name='real_image')

        self.cat = tf.placeholder(tf.float32, [None, config.cat_dim], name='category_var')
        self.c = tf.placeholder(tf.float32, [None, config.c_dim], name='continuous_var')
        self.z = tf.placeholder(tf.float32, [None, config.z_dim], name='noise_var')
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.c_cat = tf.concat([self.cat, self.c], axis=1)

        self.z_c = tf.concat([self.z, self.cat, self.c], axis=1)
        print(self.z_c.get_shape())

        # build graph
        self.G_image, _ = generator(self.z_c, config.image_channel, self.is_training)
        #   produce probabilities for real image
        self.D, self.D_logit, _, _, _, _ = discriminator_and_q(self.x, config.cat_dim, config.c_dim,
                                                            is_training=self.is_training)
        #   produce probabilities, [category and continuous] var for fake image
        self.D_fake, self.D_fake_logit, self.Qg_cat_prob, self.Qg_cat_logit, self.Qg_c, _ = discriminator_and_q(
            self.G_image, config.cat_dim, config.c_dim, reuse=True, is_training=self.is_training
        )

        # define losses
        with tf.variable_scope('d_loss'):
            self.d_loss_real = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.D),
                                                               logits=self.D_logit)
            self.d_loss_fake = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(self.D_fake),
                                                               logits=self.D_fake_logit)
            self.d_loss = self.d_loss_real + self.d_loss_fake

        with tf.variable_scope('g_loss'):
            self.g_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.D_fake),
                                                          logits=self.D_fake_logit)

        with tf.variable_scope('q_loss'):
            Q_c_cat_givenx = tf.concat([self.Qg_cat_prob, self.Qg_c], axis=1)
            self.cond_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(Q_c_cat_givenx + 1e-8)*self.c_cat, axis=1))
            self.ent = tf.reduce_mean(-tf.reduce_sum(tf.log(self.c_cat + 1e-8)*self.c_cat, axis=1))
            self.q_loss = self.cond_ent + self.ent
            #self.cat_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.cat, logits=self.Qg_cat_logit)
            #self.c_loss = tf.losses.mean_squared_error(labels=self.c, predictions=self.Qg_c)
            #self.c_ent = tf.losses.softmax_cross_entropy(onehot_labels=self.c, logits=self.Qg_c)
            #self.q_loss = self.cat_loss + self.c_loss #+ self.c_ent

        # define optimizer
        self.vars_D = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        self.vars_G = [var for var in tf.trainable_variables() if 'generator' in var.name]
        self.vars_Q = [var for var in tf.trainable_variables() if 'recognition' in var.name]
        with tf.variable_scope('opt_D'):
            self.opt_D = tf.train.AdamOptimizer(config.lr_d, beta1=config.beta1).minimize(
                self.d_loss, var_list=self.vars_D
            )

        with tf.variable_scope('opt_G'):
            self.opt_G = tf.train.AdamOptimizer(config.lr_g, beta1=config.beta1).minimize(
                self.g_loss, var_list=self.vars_G
            )

        with tf.variable_scope('opt_Q'):
            self.opt_Q = tf.train.AdamOptimizer(config.lr_d, beta1=config.beta1).minimize(
                self.q_loss
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
                    c = np.random.uniform(0, 1, size=[config.batch_size, config.c_dim])
                    #c = utils.generate_c(config.batch_size, config.c_dim)
                    cat = utils.generate_labels(config.batch_size)

                    t1 = time.time()

                    # update D network
                    _, summary_d, cur_loss_d, cur_loss_d_real, cur_loss_d_fake = sess.run(
                        [self.opt_D, self.d_summary, self.d_loss, self.d_loss_real, self.d_loss_fake],
                        feed_dict={self.z: z, self.x: image, self.c: c, self.cat: cat, self.is_training: True})
                    self.summary_writer.add_summary(summary_d, step)

                    # update G network
                    _, summary_g, cur_loss_g = sess.run([self.opt_G, self.g_summary, self.g_loss],
                                                        feed_dict={
                                                            self.z: z,
                                                            self.c: c,
                                                            self.cat: cat,
                                                            self.is_training: True
                                                        })
                    self.summary_writer.add_summary(summary_g, step)

                    # update Q network
                    _, summary_q, cur_loss_ent, cur_q_loss, cur_loss_cond_ent, cur_cat_logit, cur_Qg_c = sess.run(
                        [self.opt_Q, self.q_summary, self.ent, self.q_loss, self.cond_ent, self.Qg_cat_logit, self.Qg_c],
                        feed_dict={self.z: z, self.c: c, self.cat: cat, self.is_training: True}
                    )
                    self.summary_writer.add_summary(summary_q, step)

                    t2 = time.time()

                    batch_num = batch_num + 1

                    if step % config.summary_every_n_steps == 0:
                        print("Finished {}, Time:{:.2f}s, d_loss: {:.8f}, g_loss: {:.8f}, | H: {:.8f} and q_cross_ent: {:.8f} | q_loss: {:.8f}".
                              format(step, t2 - t1, cur_loss_d, cur_loss_g, cur_loss_ent, cur_loss_cond_ent, cur_q_loss))

                    if step in [3, 4, 5, 6, 7]:
                        print('cat: ', cur_cat_logit)
                        print('c', cur_Qg_c)

                    if step % config.sample_every_n_steps == 0:
                        z_ = utils.generate_z(81, config.z_dim)
                        cat_ = utils.sample_label(81, 9)
                        c_ = np.random.uniform(0, 1, size=[81, config.c_dim])
                        #c_ = utils.generate_c(81, config.c_dim)
                        samples, summary_image = sess.run([self.G_image, self.image_summary],
                                                          feed_dict={self.z: z_, self.cat: cat_, self.c: c_, self.is_training: False})
                        self.summary_writer.add_summary(summary_image, step)
                        # utils.save_images(np.reshape(samples, newshape=[64, 28, 28]), [8, 8], config.sampledir + '/{}.png'.format(step))
                        fig = utils.plot(samples, 9, [28, 28])
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

'''
    lcat_sample = np.reshape(np.array([e for e in range(10) for _ in range(10)]),[100,1])
    a = a = np.reshape(np.array([[(e/4.5 - 1.)] for e in range(10) for _ in range(10)]),[10,10]).T
    b = np.reshape(a,[100,1])
    c = np.zeros_like(b)
    lcont_sample = np.hstack([b,c])
    
    Constructing latent variables
    will be used in image interpolations
    
    ## reload the model
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess,ckpt.model_checkpoint_path)
'''