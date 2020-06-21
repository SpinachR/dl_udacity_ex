import os
from functools import partial
import logging
import math
from utils import *
import scipy.misc
import losses
from tf_cv import *
from tf_ops import *
import argparse
import dataset_utils
from tensorflow.python.framework import dtypes

fprint = partial(print, flush=True)

def resnet_generator(z,
                     cat,
                     params=None,
                     is_training=True,
                     reuse=None,
                     scope='generator',
                     ngf=64,
                     updates_collections='not_update_ops'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME', activation_fn=None,
                            weights_initializer=tf.variance_scaling_initializer):
            with slim.arg_scope([slim.batch_norm], decay=params['batch_norm_decay'], center=True, scale=True,
                                is_training=is_training, updates_collections=updates_collections):
                net = tf.concat([z, cat], axis=1)
                net = slim.fully_connected(net, 4*4*512, activation_fn=None, weights_initializer=tf.variance_scaling_initializer)
                net = tf.reshape(net, [-1, 4, 4, 512])
                
                net = upResidualBlock(net, input_dim=512, output_dim=256, kernel_size=3)               
                net = upResidualBlock(net, input_dim=256, output_dim=128, kernel_size=3)          
                net = upResidualBlock(net, input_dim=128, output_dim=64, kernel_size=3)
                
                ## following improved_wgan
                net = tf.nn.relu(slim.batch_norm(net))
                ##
                
                # net = concat_img_vec(net, cat)
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), mode='REFLECT')
                net = slim.conv2d(net, 3, kernel_size=3, stride=1, padding='VALID', activation_fn=None, normalizer_fn=None)
                
                return tf.nn.tanh(net)
                
                
class Label2ImageDA(object):
    def __init__(self, params):
        self.params = params
        self.G = partial(resnet_generator, params=params)
        
    def create_model(self, z_cat, noise):
        params = self.params
        ######################################################################
        # Build Model
        ######################################################################
        
        ################
        # generator ####
        ################
        cat_var = tf.one_hot(z_cat, depth=params['cat_dims'])
        self.gen_images = self.G(noise, cat_var, reuse=None, scope='generator')
        
        
params = {
    'epoches': 200,
    'batch_norm_decay': 0.9,
    'lrelu_leakiness': 0.2,
    'num_classes': 10,
    'cat_dims': 10,
    'noise_dims': 128,

    'settings': 'class_conditional_generation_',
    'pretrained_model': None,
    'max_to_keep': 5,
    'gan_type': 'DCGAN',

    'batch_size': 32,  # batch size
    'log_dir': 'log',  # path for summary file
    'checkpoint_dir': 'checkpoint',

    'lr': 0.001,
    'lambda_d': 0.01,  # trade-off between source_task_loss and discrepancy loss
    'lambda_s': 1,
    'lambda_t': 0.01,

    'mi_increasemental': 0.001,

    #########################
    # D Hyperparameters    ##
    #########################
    'ndf': 64,
    'noise_std': 0.1,
    'n_extra_layers_d': 0,
    'discriminator_dropout_keep_prob': 1,
    'discriminator_noise_stddev': 0,
    'discriminator_kernel_size': 4,
    
    'n_extra_layers_g': 2,

    # augmentation
    'no_aug': True,
    'intens_scale': (0.25, 1.5),
    'intens_offset': (-0.5, 0.5),
    'gaussian_noise_std': 0.15,
    'intens_flip': True,

    'random_brightness': 0.2,
    'random_saturation': (0.5, 1.5),
    'random_hue': 0.2,
    'random_contrast': (0.5, 1.5),
    
    'd_mutual_hyper': 0.1,
    't_vat_hyper': 1,

}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', default='mnist_m', help='')
    opt = parser.parse_args()
    print(opt)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    params['settings'] = params['settings'] + opt.target_dataset

    params['log_dir'] = os.path.join(params['settings'], params['log_dir'])
    params['checkpoint_dir'] = os.path.join(params['settings'], params['checkpoint_dir'])
    sourceOnly_dir = os.path.join(params['settings'], 'sourceOnly_checkpoints')
    if not os.path.exists(params['log_dir']):
        os.makedirs(params['log_dir'])
    if not os.path.exists(params['checkpoint_dir']):
        os.makedirs(params['checkpoint_dir'])
        
    z_cat = tf.placeholder(tf.int32, shape=[None], name='category_code')
    noise = tf.placeholder(tf.float32, shape=[None, params['noise_dims']], name='noise')
    model = Label2ImageDA(params)
    model.create_model(z_cat, noise)
    
    
    
    #### sample real data
    # target_data = dataset_utils.load_svhn('datasets/svhn', dtype=dtypes.uint8, one_hot=False)
    # X = target_data.train.images
    # Y = target_data.train.labels
    # real_samples = []
    
    # for i in range(10):
        # j = np.random.randint(100)
        # real_samples.append(X[Y==i][j])
        
    # R_I = merge_one(np.asarray(real_samples), 10, 1)
    # path1 = os.path.join(params['log_dir'], opt.target_dataset + 'real0-9.jpg')
    # save_images(R_I, path1)
    
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        
        gen_variables = slim.get_variables(scope="generator")
        if opt.target_dataset == 'mnist_m':
            model_path = 'mnist2mnist-m_smallD_C3(std0.1)_ResG(updatedBN_RELU)_(0.3mi)_Aug_False_ganDCGAN_dmutual_0.1_tvat_1/checkpoint/model.ckpt-160'
        elif opt.target_dataset == 'smnist': # svhn 2 mnist
            model_path = 'svhn2mnist_smallD_C3(std0.1)_ResG(updatedBN_RELU)_(0.3mi)_Aug_False_ganDCGAN_dmutual_0.1_tvat_1/checkpoint/model.ckpt-150'
        elif opt.target_dataset == 'usps': # mnist 2 usps
            model_path = 'mnist2usps_smallD_C3(std0.1)_ResG(updatedBN_RELU)_(0.5mi)_Aug_False_ganDCGAN_dmutual_0.1_tvat_1/checkpoint/model.ckpt-150'
        elif opt.target_dataset == 'msvhn': # mnist 2 svhn
            model_path = 'mnist2svhn_smallD_C3(std0.1)_ResG(updatedBN_RELU)_(0.1mi)_IN_Aug_True_ganDCGAN_dmutual_0.1_tvat_1/checkpoint/model.ckpt-100'
        elif opt.target_dataset == 'umnist': # usps 2 mnist
            model_path = 'usps2mnist_smallD_C3(std0.1)_ResG(updatedBN_RELU)_(0.3mi)_Aug_False_ganDCGAN_dmutual_0.1_tvat_1/checkpoint/model.ckpt-150'
        elif opt.target_dataset == 'smnist-novat': # svhn 2 mnist no vat
            model_path = 'svhn2mnist_smallD_C3(std0.1)_ResG(updatedBN_RELU)_(0.3mi)_isVAT_False_ganDCGAN_dmutual_0.1_tvat_0/checkpoint/model.ckpt-150'
        elif opt.target_dataset == 'msvhn-novat': # mnist 2 svhn no vat
            model_path = 'mnist2svhn_smallD_C3(std0.1)_ResG(updatedBN_RELU)_(0.1mi_5g)_IN__isVAT_False_ganDCGAN_dmutual_0.1_tvat_1/checkpoint/model.ckpt-150'
        
        else:
            raise NotImplementedError
        
        saver = tf.train.Saver(gen_variables)
        saver.restore(sess, model_path)
        fprint('load successfully...', opt.target_dataset)
        
        noise_ = np.random.uniform(-1., 1, size=[25, params['noise_dims']])
        z = np.random.randint(10, size=25)
        fd = {
            noise: noise_,
            z_cat: np.arange(10), # 5*np.ones(100)
        }
        
        
        
        # For linear interpolation
        
        for i in range(10):
            num_interpolations = 20
            z = np.repeat(np.arange(10), num_interpolations)
            
            noise1 = np.repeat(np.random.uniform(-1., 1, size=(10,params['noise_dims'])), num_interpolations, axis=0)
            noise2 = np.repeat(np.random.uniform(-1., 1, size=(10,params['noise_dims'])), num_interpolations, axis=0)
            alpha = np.tile(np.arange(num_interpolations) * 1.0 / (num_interpolations-1), 10)
            alpha = alpha.reshape(-1,1)
            noise_ = np.float32((1-alpha)*noise1+alpha*noise2)
            
            fd = {
                noise: noise_,
                z_cat: z
            }
            
            s_I = sess.run(model.gen_images, fd)
            s_Images = merge_one(s_I, num_interpolations, 10)
            
            path1 = os.path.join(params['log_dir'], opt.target_dataset + 'gen_interpolation_%d.jpg'%i)
            save_images(s_Images, path1)
        
        

if __name__ == '__main__':
    main()
            
        
        
    
                
                

