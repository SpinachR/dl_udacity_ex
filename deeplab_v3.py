from PIL import Image
import os
from functools import partial
import logging
import math
from utils import *
import scipy.misc
import losses
from cityscapes import inputs
import cityscapes
import synthia
from tf_cv import *
from tf_ops import *
import six


fprint = partial(print, flush=True)
IMG_MEAN = [104.00698793,116.66876762,122.67891434]


# blocks
def _start_block(inputs):
    outputs = _conv2d(inputs, 7, 64, 2, name='conv1')
    outputs = _batch_norm(outputs, name='bn_conv1', is_training=False, activation_fn=tf.nn.relu)
    outputs = _max_pool2d(outputs, 3, 2, name='pool1')
    return outputs


def _bottleneck_resblock(x, num_o, name, half_size=False, identity_connection=True):
    first_s = 2 if half_size else 1
    assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
    # branch1
    if not identity_connection:
        o_b1 = _conv2d(x, 1, num_o, first_s, name='res%s_branch1' % name)
        o_b1 = _batch_norm(o_b1, name='bn%s_branch1' % name, is_training=False, activation_fn=None)
    else:
        o_b1 = x
    # branch2
    o_b2a = _conv2d(x, 1, num_o / 4, first_s, name='res%s_branch2a' % name)
    o_b2a = _batch_norm(o_b2a, name='bn%s_branch2a' % name, is_training=False, activation_fn=tf.nn.relu)

    o_b2b = _conv2d(o_b2a, 3, num_o / 4, 1, name='res%s_branch2b' % name)
    o_b2b = _batch_norm(o_b2b, name='bn%s_branch2b' % name, is_training=False, activation_fn=tf.nn.relu)

    o_b2c = _conv2d(o_b2b, 1, num_o, 1, name='res%s_branch2c' % name)
    o_b2c = _batch_norm(o_b2c, name='bn%s_branch2c' % name, is_training=False, activation_fn=None)
    # add
    outputs = _add([o_b1, o_b2c], name='res%s' % name)
    # relu
    outputs = _relu(outputs, name='res%s_relu' % name)
    return outputs


def _dilated_bottle_resblock(x, num_o, dilation_factor, name, identity_connection=True):
    assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
    # branch1
    if not identity_connection:
        o_b1 = _conv2d(x, 1, num_o, 1, name='res%s_branch1' % name)
        o_b1 = _batch_norm(o_b1, name='bn%s_branch1' % name, is_training=False, activation_fn=None)
    else:
        o_b1 = x
    # branch2
    o_b2a = _conv2d(x, 1, num_o / 4, 1, name='res%s_branch2a' % name)
    o_b2a = _batch_norm(o_b2a, name='bn%s_branch2a' % name, is_training=False, activation_fn=tf.nn.relu)

    o_b2b = _dilated_conv2d(o_b2a, 3, num_o / 4, dilation_factor, name='res%s_branch2b' % name)
    o_b2b = _batch_norm(o_b2b, name='bn%s_branch2b' % name, is_training=False, activation_fn=tf.nn.relu)

    o_b2c = _conv2d(o_b2b, 1, num_o, 1, name='res%s_branch2c' % name)
    o_b2c = _batch_norm(o_b2c, name='bn%s_branch2c' % name, is_training=False, activation_fn=None)
    # add
    outputs = _add([o_b1, o_b2c], name='res%s' % name)
    # relu
    outputs = _relu(outputs, name='res%s_relu' % name)
    return outputs


def _ASPP(x, depth, dilations):

    inputs_size = tf.shape(x)[1:3]
    o = []
    
    conv_1_1 = tf.nn.relu(_conv2d(x, 1, depth, stride=1, name='fc_1_1', biased=True))
    conv_3_3_1 = tf.nn.relu(_dilated_conv2d(x, 3, depth, dilation_factor=dilations[0], name='fc_3_3_1', biased=True))
    conv_3_3_2 = tf.nn.relu(_dilated_conv2d(x, 3, depth, dilation_factor=dilations[1], name='fc_3_3_2', biased=True))
    conv_3_3_3 = tf.nn.relu(_dilated_conv2d(x, 3, depth, dilation_factor=dilations[2], name='fc_3_3_3', biased=True))
    
    image_level_features = tf.reduce_mean(x, [1, 2], name='global_average_pooling', keepdims=True)
    image_level_features = tf.nn.relu(_conv2d(image_level_features, 1, depth, stride=1, name='fc_global_1_1', biased=True))
    image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample', align_corners=True)
    
    net = tf.concat([conv_1_1, conv_3_3_1, conv_3_3_2, conv_3_3_3, image_level_features], axis=3, name='concat')
    net = tf.nn.relu(_conv2d(net, 1, depth, stride=1, name='fc_1_1_concat', biased=True))
    
    return net

# layers
def _conv2d(x, kernel_size, num_o, stride, name, biased=False):
    """
    Conv2d without BN or relu.
    """
    num_x = x.shape[3].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        s = [1, stride, stride, 1]
        o = tf.nn.conv2d(x, w, s, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o


def _dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, biased=False):
    """
    Dilated conv2d without BN or relu.
    """
    num_x = x.shape[3].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o


def _relu(x, name):
    return tf.nn.relu(x, name=name)


def _add(x_l, name):
    return tf.add_n(x_l, name=name)


def _max_pool2d(x, kernel_size, stride, name):
    k = [1, kernel_size, kernel_size, 1]
    s = [1, stride, stride, 1]
    return tf.nn.max_pool(x, k, s, padding='SAME', name=name)


def _batch_norm(x, name, is_training, activation_fn, trainable=True):
    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances) frozen,
    # and to not update the values provided by the pre-trained model by setting is_training=False.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.
    # Set trainable = False to remove them from trainable_variables.
    # we tend to train the gamma and beta...
    with tf.variable_scope(name) as scope:
        o = tf.contrib.layers.batch_norm(
            x,
            scale=True,
            activation_fn=activation_fn,
            is_training=is_training,
            trainable=trainable,
            scope=scope)
        return o
        
        
def _upscore_layer(bottom, shape,
                   num_classes, name, debug=False,
                   ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.stack(new_shape)

        logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
        f_shape = [ksize, ksize, num_classes, in_features]

        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

        if debug:
            deconv = tf.Print(deconv, [tf.shape(deconv)],
                              message='Shape of %s' % name,
                              summarize=4, first_n=1)

    return deconv


def get_deconv_filter(f_shape):
    width = f_shape[0]
    height = f_shape[1]
    f = math.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="up_filter", initializer=init,
                          shape=weights.shape)
    return var


def deeplab_v3(inputs, num_classes, phase, reuse, scope='deeplab'):
    '''
    :param inputs: images
    :param num_classes:
    :param phase: train (True) or test (False) for BN layers in the decoder
    :return: segmentation results. [N, H, W, num_classes]
    '''
    
    inputs_size = tf.shape(inputs)[1:3]
    
    inputs = (inputs + 1) * 127.5

    red, green, blue = tf.split(value=inputs, axis=3, num_or_size_splits=3)
    bgr = tf.concat(axis=3, values=[
        blue - IMG_MEAN[0],
        green - IMG_MEAN[1],
        red - IMG_MEAN[2],
    ])
    inputs = bgr
    
    with tf.variable_scope(scope, reuse=reuse) as scope:
        outputs = _start_block(inputs)
        print("after start block:", outputs.shape)     # 128, 256  /4

        outputs = _bottleneck_resblock(outputs, 256, '2a', identity_connection=False)
        outputs = _bottleneck_resblock(outputs, 256, '2b')
        outputs = _bottleneck_resblock(outputs, 256, '2c')
        print("after block1:", outputs.shape)          # 128, 256

        outputs = _bottleneck_resblock(outputs, 512, '3a', half_size=True, identity_connection=False)
        for i in six.moves.range(1, 4):
            outputs = _bottleneck_resblock(outputs, 512, '3b%d' % i)
        print("after block2:", outputs.shape)          # 64, 128   /8

        outputs = _dilated_bottle_resblock(outputs, 1024, 2, '4a', identity_connection=False)
        for i in six.moves.range(1, 23):
            outputs = _dilated_bottle_resblock(outputs, 1024, 2, '4b%d' % i)
        print("after block3:", outputs.shape)          # 64, 128   /8

        outputs = _dilated_bottle_resblock(outputs, 2048, 2, '5a', identity_connection=False)
        outputs = _dilated_bottle_resblock(outputs, 2048, 4, '5b')
        outputs = _dilated_bottle_resblock(outputs, 2048, 8, '5c')

        encoding = outputs
        print("before ASPP:", outputs.shape)
        # decoding
        outputs = _ASPP(encoding, 256, [12, 24, 36])
        
        outputs = _conv2d(outputs, 1, num_classes, stride=1, name='fc_last', biased=True)
        # outputs = tf.image.resize_bilinear(outputs, inputs_size, name='last_resize', align_corners=True)
        
        outputs = _upscore_layer(outputs, shape=tf.shape(inputs), num_classes=num_classes, name='fc_deconv', ksize=16, stride=8)
        
        return outputs
        
        
   
def build_resnet_block(inputres, dim, name="resnet", padding_mode="REFLECT"):
    with tf.variable_scope(name):
        with slim.arg_scope([slim.conv2d], padding='VALID', kernel_size=3, stride=1, activation_fn=None,
                            weights_initializer=tf.variance_scaling_initializer):
            with slim.arg_scope([instance_norm], shift=True, scale=True):
                out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding_mode)
                out_res = tf.nn.relu(instance_norm(slim.conv2d(out_res, dim)))

                out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding_mode)
                out_res = instance_norm(slim.conv2d(out_res, dim))  # no relu here
                return out_res + inputres  # sometimes, there is no relu here

                
def generator_resnet(inputgen,
                     noise,
                     params,
                     is_training=True,
                     reuse=None,
                     scope='generator',
                     ngf=64,
                     num_blocks=16,
                     skip=False,
                     updates_collections='not_update_ops'):
    # if noise is None:
        # noise = tf.random_uniform(tf.shape(inputgen), minval=-1., maxval=1.)
        
    # inputgen = tf.one_hot(tf.squeeze(inputgen, axis=3), params['num_classes'], on_value=1.0, off_value=0.0)
    
    # proj_shape = inputgen.shape.as_list()[:3] + [1]
    
    # inputgen = tf.concat([inputgen, noise], axis=3)
    fprint('Input (source images) to generator: ', inputgen.get_shape())

    padding = 'REFLECT'
    pad_input = tf.pad(inputgen, [[0, 0], [3, 3], [3, 3], [0, 0]], padding)
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=None, weights_initializer=tf.variance_scaling_initializer):
            with slim.arg_scope([instance_norm], shift=True, scale=True):
                net = tf.nn.relu(instance_norm(
                    slim.conv2d(pad_input, ngf, kernel_size=7, stride=1, padding='VALID', activation_fn=None)))
                net = tf.nn.relu(instance_norm(
                    slim.conv2d(net, ngf * 2, kernel_size=3, stride=2, padding='SAME', activation_fn=None)))
                net = tf.nn.relu(instance_norm(
                    slim.conv2d(net, ngf * 4, kernel_size=3, stride=2, padding='SAME', activation_fn=None)))

                for i in range(num_blocks):
                    net = build_resnet_block(net, ngf * 4, name='resnet_g_%d' % (i + 1), padding_mode=padding)

                net = tf.nn.relu(instance_norm(upsample(net, ngf * 2, kernel_size=3, stride=2, method='resize_conv')))
                net = tf.nn.relu(instance_norm(upsample(net, ngf, kernel_size=3, stride=2, method='resize_conv')))

                if skip is True:
                    net = tf.concat(values=[inputgen, net], axis=3)

                net = tf.pad(net, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), mode='REFLECT')
                net = slim.conv2d(net, 3, kernel_size=7, stride=1, padding='VALID', activation_fn=None)

                out_gen = tf.nn.tanh(net)

                return out_gen


params = {
    'epoches': 26,
    'batch_norm_decay': 0.99,
    'lrelu_leakiness': 0.2,
    'weight_decay': 0.0005,

    'num_classes': 16,
    'num_examples_trainset': 9400,  # remove the samples with different size
    'num_examples_testset': 500,
    'upsample_logits': True,

    'height': 512,  # 320,  # Small Image Size could save a lot memory
    'width': 1024,  # 640,

    'cat_dims': 9,
    'noise_dims': 128,
    
    'deeplab_init': 'datasets/resnet_path/deeplab_resnet_101_better.ckpt',

    'settings': 'deeplab_v3(2grid)_deconvup_synthia2cityscapes_nol2sp_swa3_',
    'pretrained_model': 'deeplab_v3_synthia2cityscapes_sourceonly/no_l2sp_head_248/model.ckpt-3', #'deeplab_sourceOnly_upsample19/model.ckpt-6',
    'generator_model': 'synthia2cityscape_resnet_seg_vgg_1024_GAN-SN_vgggen_bs1_optadam_numResiblocks16_GANType_GAN-SN/checkpoint/model.ckpt-15',#'synthia2cityscape_resnet_seg_vgg_1024_GAN-SN_vgggen_bs1_optadam_numResiblocks16_GANType_GAN-SN/checkpoint/model_0.38868940636329.ckpt-11',#'synthia2cityscape_resnet_seg_vgg_1024_GAN-SN_vgggen_bs1_optadam_numResiblocks16_GANType_GAN-SN/checkpoint/model.ckpt-3',
    'max_to_keep': 20,

    'resolution': 512,  # for width. Rescale with the same ratio

    'batch_size': 1,  # batch size
    'log_dir': 'log',  # path for summary file
    'checkpoint_dir': 'checkpoint_hd',

    'sgd': False,  # using sgd-momentum or adam
    'base_lr': 2.5e-4,

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

    'n_extra_layers_g': 0,

    # augmentation
    'no_aug': False,
    'intens_scale': None,  # (0.25, 1.5),
    'intens_offset': None,  # (-0.5, 0.5),
    'gaussian_noise_std': 0,  # 0.15,
    'intens_flip': False,

    'random_brightness': None,  # 0.2,
    'random_saturation': None,  # (0.5, 1.5),
    'random_hue': None,  # 0.2,
    'random_contrast': None,  # (0.5, 1.5),

    # For Segmentation
    'hflip': True,
    'random_resize_crop': True,
    'fine_size': 512,

    # large hyper
    'w_mi': 0.001,
    'w_tvat': 0,

    'g_vgg_weight': 1e-5,
    'g_l1_weight': 1,
    'g_gan_weight': 1,
}


class SegGAN(object):
    def __init__(self, params, is_training):
        self.params = params
        self.is_training = is_training
        self.res_seg = partial(deeplab_v3, num_classes=params['num_classes'], phase=is_training)

    def create_model(self, target_images, lambda_s_vat, lambda_t_vat, lr, target_labels=None, source_images=None,
                     source_labels=None):
        params = self.params

        ################
        # pre-process ##
        ################

        # source_images = tf.image.convert_image_dtype(source_images, tf.float32)
        source_images = tf.image.convert_image_dtype(source_images, tf.float32)
        target_images = tf.image.convert_image_dtype(target_images, tf.float32)
        
        # do augmentation if necessary
        if params['hflip']:
            print('random flip input ...', flush=True)
            uniform_random = tf.random_uniform([], 0, 1.0, seed=None)     
            do_a_flip_random = tf.greater(uniform_random, 0.5)
            source_images = tf.cond(do_a_flip_random, lambda: tf.image.flip_left_right(source_images), lambda: source_images)
            source_labels = tf.cond(do_a_flip_random, lambda: tf.image.flip_left_right(source_labels), lambda: source_labels)
        

        source_images1 = tf.image.resize_images(source_images, size=(params['height'], params['width']),
                                                method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        source_labels1 = tf.image.resize_images(source_labels, size=(params['height'], params['width']),
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

        target_images1 = tf.image.resize_images(target_images, size=(params['height'], params['width']),
                                                method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        target_labels1 = tf.image.resize_images(target_labels, size=(params['height'], params['width']),
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

        # scale to [-1, 1]
        source_images1 = (source_images1 - 0.5) * 2.0
        target_images1 = (target_images1 - 0.5) * 2.0

        self.target_images1 = target_images1
        self.target_labels1 = target_labels1
        
        self.source_images1 = source_images1

        ######################################################################
        # Build Model
        ######################################################################
        # generator
        gen_images = generator_resnet(source_images1, params=params, noise=None, reuse=False, scope='generator', is_training=True)
            
        self.gen_images = gen_images
        fprint('gen_images shape:', self.gen_images.get_shape())
        
        # segmentor
        target_logits = self.res_seg(target_images1, reuse=None)  # scope resnet_v1_101
        source_logits = self.res_seg(source_images1, reuse=True)
        gen_logits = self.res_seg(self.gen_images, reuse=True)
        
        # load
        restore_var = [v for v in tf.global_variables() if ('deeplab' in v.name) and ('fc' not in v.name)]
        self.loader = tf.train.Saver(var_list=restore_var)

        ######################################################################
        # Define losses
        ######################################################################

        ######################################################################
        # Define losses
        ######################################################################
        
        
        ## apply L2-SP regularization
        # reader = tf.train.NewCheckpointReader(params['deeplab_init'])
        # l2_losses = []

        # only weights
        # weight_vars = [v for v in restore_var if ('weights' in v.name)]
        # fprint(len(weight_vars))
        # for v in weight_vars:
            # print(v.name)
            # pre_trained_weights = reader.get_tensor(v.name.split(':')[0])
            # l2_losses.append(tf.nn.l2_loss(v - pre_trained_weights))

        # self.l2_sp = tf.add_n(l2_losses)
        
        
        head = np.asarray([1, 2, 1, 5, 5, 2, 5, 4, 1, 1, 2, 4, 1, 3, 4, 2])
        fprint('Apply loss weights ...')
        fprint(head)
        
        self.t_task_loss = losses.robust_seg_loss(target_logits, target_labels1, num_classes=params['num_classes'], head=head)
        self.s_task_loss = losses.robust_seg_loss(source_logits, source_labels1, num_classes=params['num_classes'], head=head)  
        self.gen_task_loss = losses.robust_seg_loss(gen_logits, source_labels1, num_classes=params['num_classes'], head=head)


        
        self.seg_loss = self.s_task_loss + 0.1*self.gen_task_loss# + 1e-4*self.l2_sp  #

        ######################################################################
        # construct train op
        ######################################################################
        t_vars = tf.trainable_variables()
        seg_optimizer = tf.train.AdamOptimizer(0.00001, beta1=0.9)
        seg_vars = [var for var in t_vars if 'deeplab' in var.name]
        self.seg_op = slim.learning.create_train_op(self.seg_loss, seg_optimizer, variables_to_train=seg_vars)

        ######################################################################
        # For Visualize the pred
        ######################################################################
        self.gt_vis = decode_labels(tf.squeeze(target_labels1, axis=3), params['num_classes'] + 1)

        target_pred = tf.argmax(target_logits, axis=3)
        self.pred_vis = decode_labels(target_pred, params['num_classes'])
        
        # source
        gen_pred = tf.argmax(gen_logits, axis=3)
        source_pred = tf.argmax(source_logits, axis=3)
        self.gen_pred_vis = decode_labels(gen_pred, params['num_classes'])
        self.src_pred_vis = decode_labels(source_pred, params['num_classes'])
        self.src_gt_vis = decode_labels(tf.squeeze(source_labels1, axis=3), params['num_classes'] + 1)

    def eval_model(self, test_images, test_labels, swa_n):
        params = self.params
        ################
        # pre-process ##
        ################
        test_images = tf.image.convert_image_dtype(test_images, tf.float32)
        ref_test_images = tf.image.resize_images(test_images, size=(params['height'], params['width']),
                                             method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        ref_test_images = (ref_test_images - 0.5) * 2.0
        self.test_images = ref_test_images
        self.test_labels = test_labels
        self.small_test_labels = tf.image.resize_images(test_labels, size=(params['height'], params['width']),
                                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

        test_logits =self.res_seg(ref_test_images, reuse=True)
        
        # test_pred1: fused normal size
        self.test_pred1 = tf.argmax(test_logits, dimension=3)
        
        # test_pred3: fused upsampled size
        test_logits3 = tf.image.resize_bilinear(test_logits, size=(1024, 2048), align_corners=True)
        self.test_pred3 = tf.argmax(test_logits3, dimension=3)
        
        
        ###################################
        # swa
        ###################################
        _x = self.res_seg(ref_test_images, reuse=False, scope='seg_swa')  # create another model
        
        seg_swa = [var for var in tf.global_variables() if 'seg_swa' in var.name]
        seg = [var for var in tf.global_variables() if 'deeplab' in var.name]
        
        self.swa_update_op = assign_moving_average(seg_swa, seg, 1.0 / (swa_n))
        
        test_logits_swa = self.res_seg(ref_test_images, reuse=True, scope='seg_swa')
        # test_pred_swa1: fused normal size
        self.test_pred_swa1 = tf.argmax(test_logits_swa, dimension=3)

        



def main():
    n_iter_per_epoch = params['num_examples_trainset'] // params['batch_size']
    n_iter_per_epoch_val = math.ceil(1.0 * params['num_examples_testset'] / 1.)
    num_training_steps = params['epoches'] * n_iter_per_epoch

    if params['sgd']:
        opt = 'sgd'
    else:
        opt = 'adam'

    params['settings'] = params['settings'] + '_bs' + str(params['batch_size']) + '_opt' + opt
    params['log_dir'] = os.path.join(params['settings'], params['log_dir'])
    params['checkpoint_dir'] = os.path.join(params['settings'], params['checkpoint_dir'])

    if not os.path.exists(params['log_dir']):
        os.makedirs(params['log_dir'])
    if not os.path.exists(params['checkpoint_dir']):
        os.makedirs(params['checkpoint_dir'])

    with tf.device('/cpu:0'):
        src_img_batch, src_label_batch = synthia.inputs('datasets/synthia/synthia.tfrecord', params, is_training=False, shuffle=True)
        trg_img_batch, trg_label_batch = cityscapes.inputs('datasets/cityscapes/train16_cityscape.tfrecord', params,
                                                           is_training=False, shuffle=True)
        test_img_batch, test_label_batch = cityscapes.inputs('datasets/cityscapes/test16_cityscape.tfrecord', params,
                                                             is_training=False, shuffle=False)
        

    is_training = tf.placeholder(tf.bool, name='is_training')
    cur_step = tf.placeholder(tf.int64, name='current_steps')
    lambda_s_vat = tf.placeholder(tf.float32)

    learning_rate = tf.scalar_mul(params['base_lr'], tf.pow((1 - cur_step / 59500), 0.9))
    swa_n = tf.placeholder(tf.float32)
    
    model = SegGAN(params, is_training)

    model.create_model(trg_img_batch, lambda_s_vat=lambda_s_vat, lambda_t_vat=None, lr=learning_rate,
                       target_labels=trg_label_batch,
                       source_images=src_img_batch, source_labels=src_label_batch)

    model.eval_model(test_img_batch, test_label_batch, swa_n=swa_n)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    _acc1 = []
    _mIU1 = []
    _acc2 = []
    _mIU2 = []
    _acc3 = []
    _mIU3 = []
    _acc4 = []
    _mIU4 = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        model_saver = tf.train.Saver(max_to_keep=params['max_to_keep'])
        if params['pretrained_model'] is not None:
            fprint('load pretrained model ', params['pretrained_model'], ' ...')
            segVars = slim.get_variables(scope='deeplab')
            variables_to_restore = [v for v in segVars if 'fc_deconv' not in v.name]    
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, params['pretrained_model'])
            # model.loader.restore(sess, params['pretrained_model'])
            
            fprint('load pretrained generator ', params['generator_model'], ' ...')
            genVars = slim.get_variables(scope='generator')
            variables_to_restore = genVars
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, params['generator_model'])
            
            
            fprint('load successfully...')

        fprint('[*] start training ...')
        fprint('num of epoches: [%d], training steps: [%d]' % (params['epoches'], num_training_steps))
        fprint('num of iterations per epoch for test: ', n_iter_per_epoch_val)

        iter_counter = 0
        swa_num = 1.
        bestIU = 0.39
        gen_task_weight = 0
        for epoch in range(params['epoches']):
            
            batch_index = 0
            
            conf_m1 = np.zeros((params['num_classes'], params['num_classes']))
            conf_swa1 = np.zeros((params['num_classes'], params['num_classes']))
            for i in range(500):
                test_labels, test_pred, test_pred_swa1 = sess.run(
                    [model.small_test_labels, model.test_pred1, model.test_pred_swa1],
                    feed_dict={is_training: False})
                    
                flat_label = np.squeeze(test_labels, axis=3).flatten()
                flat_pred = test_pred.flatten()
                conf_m1 += fast_hist(flat_label, flat_pred, params['num_classes'])
                conf_swa1 += fast_hist(flat_label, test_pred_swa1.flatten(), params['num_classes'])

            acc1, acc_cls1, mIU1, iu1, fwav = compute_score(conf_m1)
            acc_swa1, acc_cls_swa1, mIU_swa1, iu_swa1, fwav_swa1 = compute_score(conf_swa1)
            
            fprint('#######################################################')
            fprint('Epoch [%4d]/[%4d]' % (epoch, params['epoches']))
            fprint('test_acc: [%.4f], test_acc_cls: [%.4f], test_fwav: [%.4f] test_mIU: [%.4f]' % (acc1, acc_cls1, fwav, mIU1))
            fprint('test_acc_swa: [%.4f], test_acc_cls_swa: [%.4f], test_fwav_swa: [%.4f] test_mIU_swa: [%.4f]' % (acc_swa1, acc_cls_swa1, fwav_swa1, mIU_swa1))
            fprint(iu1)
            fprint(iu_swa1)
            fprint()
            
            
            
            gen_task_weight = 0.1

            ################################
            # save model: checkpoint
            ################################

            while batch_index < n_iter_per_epoch:
            
                fd_train = {
                    is_training: True, 
                    cur_step: iter_counter,
                    lambda_s_vat: gen_task_weight,
                }

                sess.run(model.seg_op, feed_dict=fd_train)
                
                if np.mod(batch_index, 3000) == 0:
                    fd_test = {
                        is_training: False, 
                        cur_step: iter_counter,
                        lambda_s_vat: gen_task_weight,
                    }
                
                    
                    
                    conf_m1 = np.zeros((params['num_classes'], params['num_classes']))
                    conf_m3 = np.zeros((params['num_classes'], params['num_classes']))
                    conf_swa1 = np.zeros((params['num_classes'], params['num_classes']))
                                        
                    for i in range(500):
                    
                        test_labels, resized_test_labels, test_pred1,  test_pred3,  test_pred_swa1 = sess.run(
                            [model.test_labels, model.small_test_labels, model.test_pred1, model.test_pred3, model.test_pred_swa1],
                            feed_dict=fd_test)
                        
                        
                             
                        flat_label_small = resized_test_labels.flatten()
                        flat_label_big = np.squeeze(test_labels, axis=3).flatten()
                        
                        
                        conf_m1 += fast_hist(flat_label_small, test_pred1.flatten(), params['num_classes'])
                        conf_m3 += fast_hist(flat_label_big, test_pred3.flatten(), params['num_classes'])
                        conf_swa1 += fast_hist(flat_label_small, test_pred_swa1.flatten(), params['num_classes'])
                        

                    acc1, acc_cls1, mIU1, iu1, fwav1 = compute_score(conf_m1)
                    acc3, acc_cls3, mIU3, iu3, fwav3 = compute_score(conf_m3)
                    
                    acc_swa1, acc_cls_swa1, mIU_swa1, iu_swa1, fwav_swa1 = compute_score(conf_swa1)

                    fprint('#######################################################')
                    fprint('Epoch [%4d]/[%4d]' % (epoch, params['epoches']))
                    fprint('test_acc1: [%.4f], test_acc_cls1: [%.4f], test_fwav1: [%.4f] test_mIU1: [%.4f]' % (acc1, acc_cls1, fwav1, mIU1))
                    fprint('test_acc3: [%.4f], test_acc_cls3: [%.4f], test_fwav3: [%.4f] test_mIU3: [%.4f]' % (acc3, acc_cls3, fwav3, mIU3))
                    fprint('test_acc_swa: [%.4f], test_acc_cls_swa: [%.4f], test_fwav_swa: [%.4f] test_mIU_swa: [%.4f]' % (acc_swa1, acc_cls_swa1, fwav_swa1, mIU_swa1))
                    fprint(iu1)
                    fprint(iu_swa1)
                    fprint()

                    _acc1.append(acc1)
                    _mIU1.append(mIU1)

                    _acc3.append(acc3)
                    _mIU3.append(mIU3)
                    
                    _acc4.append(acc_swa1)
                    _mIU4.append(mIU_swa1)

                    ex_results = [_acc1, _acc3, _acc4, _mIU1, _mIU3, _mIU4]  # test_acc, test_mIU
                    path_ = os.path.join(params['log_dir'], 'ex_results.jpg')
                    plot_acc(ex_results, path_)

                    path_pkl = os.path.join(params['log_dir'], 'ex_results.pkl')
                    path_txt = os.path.join(params['log_dir'], 'ex_results.txt')
                    save_list(ex_results, path_pkl, path_txt)


                    # swa update
                    if epoch>=1 and mIU1>0.383:
                        sess.run(model.swa_update_op, feed_dict={swa_n: swa_num})
                        swa_num = swa_num+1.
                        fprint('[**]update swa:', swa_num)
                    
                    
                    
                    t_task_loss, s_task_loss, gen_task_loss = sess.run(
                                [model.t_task_loss, model.s_task_loss, model.gen_task_loss],
                                feed_dict={is_training: False, cur_step: iter_counter})
                    fprint('t_task_loss: [%.7f], s_task_loss: [%.7f], gen_task_loss: [%.7f]' % (t_task_loss, s_task_loss, gen_task_loss))
                                
                    
                    # To Do: save mdoel
                    current_mIU = max([mIU1, mIU3])
                    if current_mIU > bestIU:
                        bestIU = current_mIU
                        name = 'model_'+ str(current_mIU) + '.ckpt'
                        model_saver.save(sess, os.path.join(params['checkpoint_dir'], name), global_step=epoch)


                batch_index += 1
                iter_counter += 1
        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    main()
