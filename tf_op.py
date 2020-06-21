import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.contrib.framework import add_arg_scope, arg_scope
from tensorflow.contrib.layers import variance_scaling_initializer
from sn import spectral_normed_weight
from functools import partial


def _assign_moving_average(orig_val, new_val, momentum, name):
    with tf.name_scope(name):
        scaled_diff = (1 - momentum) * (new_val - orig_val)
        return tf.assign_add(orig_val, scaled_diff)


@add_arg_scope
def batch_norm(x, use_batchstats=False, shift=True, scale=True, momentum=0.99, eps=1e-3, internal_update=False,  
               scope=None, reuse=None):
    c = x._shape_as_list()[-1]
    ndim = len(x.shape)
    var_shape = [1] * (ndim - 1) + [c]

    with tf.variable_scope(scope, 'batch_norm', reuse=reuse):
        moving_m = tf.get_variable('mean', var_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_v = tf.get_variable('var', var_shape, initializer=tf.ones_initializer, trainable=False)

        def training():
            m, v = tf.nn.moments(x, list(range(ndim - 1)), keep_dims=True)
            update_m = _assign_moving_average(moving_m, m, momentum, 'update_mean')
            update_v = _assign_moving_average(moving_v, v, momentum, 'update_var')

            # tf.add_to_collection('update_ops', update_m)
            # tf.add_to_collection('update_ops', update_v)

            if internal_update:
                with tf.control_dependencies([update_m, update_v]):
                    output = (x - m) * tf.rsqrt(v + eps)

            else:
                output = (x - m) * tf.rsqrt(v + eps)

            return output

        def testing():
            m, v = moving_m, moving_v
            output = (x - m) * tf.rsqrt(v + eps)
            return output

        if isinstance(use_batchstats, bool):
            output = training() if use_batchstats else testing()
        else:
            output = tf.cond(use_batchstats, training, testing)

        if scale:
            output *= tf.get_variable('gamma', var_shape, initializer=tf.ones_initializer)

        if shift:
            output += tf.get_variable('beta', var_shape, initializer=tf.zeros_initializer)

        return output


@add_arg_scope
def instance_norm(x,
                  shift=True,
                  scale=True,
                  eps=1e-5,
                  scope=None,
                  reuse=None):
    # Expect a 4-D Tensor
    C = x._shape_as_list()[-1]

    with tf.variable_scope(scope, 'instance_norm', reuse=reuse):
        # Get mean and variance, normalize input
        m, v = tf.nn.moments(x, [1, 2], keep_dims=True)
        output = (x - m) * tf.rsqrt(v + eps)

        if scale:
            output *= tf.get_variable('gamma', C, initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))

        if shift:
            output += tf.get_variable('beta', C, initializer=tf.zeros_initializer)

    return output


@add_arg_scope
def gaussian_noise(inputs, std=0.1, is_training=False):
    def training():
        with tf.name_scope('gaussian_noise_layer'):
            noise = tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=std, dtype=tf.float32)
            return inputs + noise

    def testing():
        return inputs

    return tf.cond(is_training, training, testing, name='gaussian_noise')


def gaussian_dropout(inputs, rate=0.5, is_training=False):
    """Apply multiplicative 1-centered Gaussian noise.

    As it is a regularization layer, it is only active at training time.

    Arguments:
      rate: float, drop probability (as with `Dropout`).
          The multiplicative noise will have
          standard deviation `sqrt(rate / (1 - rate))`.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.

    References:
      - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        Srivastava, Hinton, et al.
        2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """

    def training():
        with tf.name_scope('training_phrase'):
            stddev = np.sqrt(rate / (1.0 - rate))
            return inputs * tf.random_normal(shape=tf.shape(inputs), mean=1.0, stddev=stddev)

    def testing():
        with tf.name_scope('testing_phrase'):
            return inputs

    return tf.cond(is_training, training, testing, name='gaussian_dropout')

    
# @add_arg_scope
# def upsample(net, num_filters, scale=2, method='resize_conv', scope=None):
    # """Performs spatial upsampling of the given features.
    # Args:
      # net: A `Tensor` of shape [batch_size, height, width, filters].
      # num_filters: The number of output filters.
      # scale: The scale of the upsampling. Must be a positive integer greater or
        # equal to two.
      # method: The method by which the features are upsampled. Valid options
        # include 'resize_conv' and 'conv2d_transpose'.
      # scope: An optional variable scope.
    # Returns:
      # A new set of features of shape
        # [batch_size, height*scale, width*scale, num_filters].
    # Raises:
      # ValueError: if `method` is not valid or
    # """
    # if scale < 2:
        # raise ValueError('scale must be greater or equal to two.')

    # with tf.variable_scope(scope, 'upsample', [net]):
        # if method == 'resize_conv':
            # net = tf.image.resize_nearest_neighbor(
                # net, [net.shape.as_list()[1] * scale,
                      # net.shape.as_list()[2] * scale],
                # align_corners=True,
                # name='resize')
            # return slim.conv2d(net, num_filters, stride=1, scope='conv')
        # elif method == 'conv2d_transpose':
            # return slim.conv2d_transpose(net, num_filters, scope='deconv')
        # else:
            # raise ValueError('Upsample method [%s] was not recognized.' % method)
@add_arg_scope
def upsample(net, num_filters, kernel_size=3, stride=2, method='resize_conv', scope=None,
             resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False):
    """Performs spatial upsampling of the given features.
    Args:
      net: A `Tensor` of shape [batch_size, height, width, filters].
      num_filters: The number of output filters.
      kernel_size:
      stride: The scale of the upsampling. Must be a positive integer greater or
        equal to two.
      method: The method by which the features are upsampled. Valid options
        include 'resize_conv' and 'conv2d_transpose'.
      scope: An optional variable scope.
      resize_method:
      align_corners:
    Returns:
      A new set of features of shape
        [batch_size, height*scale, width*scale, num_filters].
    Raises:
      ValueError: if `method` is not valid or
    """
    if stride < 2:
        raise ValueError('scale must be greater or equal to two.')

    with tf.variable_scope(scope, 'upsample', [net]):
        if method == 'resize_conv':
            net = tf.image.resize_images(net, size=[net.shape.as_list()[1] * stride, net.shape.as_list()[2] * stride],
                                         method=resize_method, align_corners=align_corners)
            net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), mode='REFLECT')
            return tf.layers.conv2d(net, num_filters, kernel_size=kernel_size, padding='valid', activation=None,
                                    kernel_initializer=tf.variance_scaling_initializer)
        elif method == 'conv2d_transpose':
            # This corrects 1 pixel offset for images with even width and height.
            # conv2d is left aligned and conv2d_transpose is right aligned for even
            # sized images (while doing 'SAME' padding).
            # https://github.com/tensorflow/models/blob/master/research/slim/nets/cyclegan.py
            net = tf.layers.conv2d_transpose(net, num_filters, kernel_size=kernel_size, strides=stride,
                                             padding='valid', activation=None,
                                             kernel_initializer=tf.variance_scaling_initializer)
            return net[:, 1:, 1:, :]
        else:
            raise ValueError('Upsample method [%s] was not recognized.' % method)
            

@add_arg_scope
def dense(x,
          num_outputs,
          scope=None,
          activation=None,
          reuse=None,
          bn=False,
          post_bn=False,
          phase=None):

    with tf.variable_scope(scope, 'dense', reuse=reuse):
        # convert x to 2-D tensor
        dim = np.prod(x._shape_as_list()[1:])
        x = tf.reshape(x, [-1, dim])
        weights_shape = (x.get_shape().dims[-1], num_outputs)

        # dense layer
        weights = tf.get_variable('weights', weights_shape,
                                  initializer=variance_scaling_initializer())
        biases = tf.get_variable('biases', [num_outputs],
                                 initializer=tf.zeros_initializer)
        output = tf.matmul(x, weights) + biases
        if bn: output = batch_norm(output, phase, scope='bn')
        if activation: output = activation(output)
        if post_bn: output = batch_norm(output, phase, scope='post_bn')

    return output

    
@add_arg_scope
def conv2d(x,
           num_outputs,
           kernel_size,
           strides,
           padding='SAME',
           activation=None,
           bn=False,
           post_bn=False,
           phase=None,
           scope=None,
           reuse=None):
    # Convert int to list
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    strides = [strides] * 2 if isinstance(strides, int) else strides

    # Convert list to valid list
    kernel_size = list(kernel_size) + [x.get_shape().dims[-1], num_outputs]
    strides = [1] + list(strides) + [1]

    # Conv operation
    with tf.variable_scope(scope, 'conv2d', reuse=reuse):
        kernel = tf.get_variable('weights', kernel_size,
                                 initializer=variance_scaling_initializer())
        biases = tf.get_variable('biases', [num_outputs],
                                 initializer=tf.zeros_initializer)
        output = tf.nn.conv2d(x, kernel, strides, padding, name='conv2d')
        output += biases
        if bn: output = batch_norm(output, phase, scope='bn')
        if activation: output = activation(output)
        if post_bn: output = batch_norm(output, phase, scope='post_bn')

    return output
    

@add_arg_scope
def conv2d_transpose(x,
                     num_outputs,
                     kernel_size,
                     strides,
                     padding='SAME',
                     output_shape=None,
                     output_like=None,
                     activation=None,
                     bn=False,
                     post_bn=False,
                     phase=None,
                     scope=None,
                     reuse=None):
    # Convert int to list
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    strides = [strides] * 2 if isinstance(strides, int) else strides

    # Convert list to valid list
    kernel_size = list(kernel_size) + [num_outputs, x.get_shape().dims[-1]]
    strides = [1] + list(strides) + [1]

    # Get output shape both as tensor obj and as list
    if output_shape:
        bs = tf.shape(x)[0]
        _output_shape = tf.stack([bs] + output_shape[1:])
    elif output_like:
        _output_shape = tf.shape(output_like)
        output_shape = output_like.get_shape()
    else:
        assert padding == 'SAME', "Shape inference only applicable with padding is SAME"
        bs, h, w, c = x._shape_as_list()
        bs_tf = tf.shape(x)[0]
        _output_shape = tf.stack([bs_tf, strides[1] * h, strides[2] * w, num_outputs])
        output_shape = [bs, strides[1] * h, strides[2] * w, num_outputs]

    # Transposed conv operation
    with tf.variable_scope(scope, 'conv2d', reuse=reuse):
        kernel = tf.get_variable('weights', kernel_size,
                                 initializer=variance_scaling_initializer())
        biases = tf.get_variable('biases', [num_outputs],
                                 initializer=tf.zeros_initializer)
        output = tf.nn.conv2d_transpose(x, kernel, _output_shape, strides,
                                        padding, name='conv2d_transpose')
        output += biases
        output.set_shape(output_shape)
        if bn: output = batch_norm(output, phase, scope='bn')
        if activation: output = activation(output)
        if post_bn: output = batch_norm(output, phase, scope='post_bn')

    return output
    

@add_arg_scope
def max_pool(x,
             kernel_size,
             strides,
             padding='SAME',
             scope=None):
    # Convert int to list
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    strides = [strides] * 2 if isinstance(strides, int) else strides

    # Convert list to valid list
    kernel_size = [1] + list(kernel_size) + [1]
    strides = [1] + list(strides) + [1]

    with tf.variable_scope(scope, 'max_pool'):
        output = tf.nn.max_pool(x, kernel_size, strides, padding=padding)

    return output

    
@add_arg_scope
def avg_pool(x,
             kernel_size=2,
             strides=2,
             padding='SAME',
             global_pool=False,
             scope=None):
    
    if global_pool:
        return tf.reduce_mean(x, axis=[1, 2])

    # Convert int to list
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    strides = [strides] * 2 if isinstance(strides, int) else strides

    # Convert list to valid list
    kernel_size = [1] + list(kernel_size) + [1]
    strides = [1] + list(strides) + [1]

    with tf.variable_scope(scope, 'avg_pool'):
        print('apply avg_pool ... ', flush=True)
        output = tf.nn.avg_pool(x, kernel_size, strides, padding=padding)

    return output
    

@add_arg_scope
def concat_img_vec(img,
                   vec,
                   scope=None):
    H, W = img._shape_as_list()[1:-1]
    V = vec._shape_as_list()[-1]

    with tf.variable_scope(scope, 'concat_img_vec'):
        # Depth-wise concatenation of vec to img
        # Replicate vec via broadcasting
        broadcast = tf.ones([1, H, W, 1])
        vec = tf.reshape(vec, [-1, 1, 1, V])
        vec = broadcast * vec
        output = tf.concat([img, vec], axis=-1)

    return output

    
@add_arg_scope
def conv2d_sn(x,
              num_outputs,
              kernel_size=4,
              stride=1,
              padding='SAME',
              scope=None,
              reuse=None,
              updates_collections=None,
              spectral_normed=True):
              
     # Convert int to list
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    strides = [stride] * 2 if isinstance(stride, int) else stride
    
    
    # Convert list to valid list
    kernel_size = list(kernel_size) + [x.get_shape().dims[-1], num_outputs]
    strides = [1] + list(strides) + [1]
    
    
    with tf.variable_scope(scope, 'conv2d_sn', reuse=reuse):
        kernel = tf.get_variable('weights', kernel_size, initializer=tf.variance_scaling_initializer)
        biases = tf.get_variable('biases', [num_outputs], initializer=tf.zeros_initializer)
        
        if spectral_normed:
            print('conv2d spectral normalization...', flush=True)
            print(updates_collections, flush=True)
            output = tf.nn.conv2d(x, spectral_normed_weight(kernel, updates_collections=updates_collections), strides=strides, padding=padding, name='conv2d_sn')
        else:
            output = tf.nn.conv2d(x, kernel, strides, padding, name='conv2d')
            
        return output+biases
        
@add_arg_scope        
def dense_sn(x,
             num_outputs,
             scope=None,
             reuse=None,
             updates_collections=None,
             spectral_normed=True):
    with tf.variable_scope(scope, 'dense_sn', reuse=reuse):
        weights_shape = (x.get_shape().dims[-1], num_outputs)
        
        # dense layer        
        weights = tf.get_variable('weights', weights_shape, initializer=tf.variance_scaling_initializer)
        biases = tf.get_variable('biases', [num_outputs], initializer=tf.zeros_initializer)
        
        if spectral_normed:
            print('dense spectral normalization...', flush=True)
            print(updates_collections, flush=True)
            output = tf.matmul(x, spectral_normed_weight(weights, updates_collections=updates_collections))
        else:
            output = tf.matmul(x, weights)
            
        return output+biases
        
        
# wgan: it has the code for weight_normalization
def ConvMeanPool(net, output_dim, kernel_size):
    output = tf.layers.conv2d(net, output_dim, kernel_size=kernel_size, padding='same',
                              kernel_initializer=tf.variance_scaling_initializer)
    output = tf.nn.avg_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return output


def downResidualBlock(net, input_dim, output_dim, kernel_size):
    conv1 = partial(tf.layers.conv2d, filters=input_dim, padding='same',
                    kernel_initializer=tf.variance_scaling_initializer)
    conv2 = partial(ConvMeanPool, output_dim=output_dim)

    # shortcut
    shortcut = ConvMeanPool(net, output_dim, kernel_size)

    #
    output = net
    output = slim.batch_norm(output)  # need the tf.variable_scope
    output = tf.nn.relu(output)
    output = conv1(output, kernel_size=kernel_size)

    output = slim.batch_norm(output)
    output = tf.nn.relu(output)
    output = conv2(output, kernel_size=kernel_size)

    return shortcut + output
    
def upResidualBlock(net, input_dim, output_dim, kernel_size):
    print('updated in 2018/9/10 ...')
    conv1 = partial(upsample, num_filters=output_dim)
    conv2 = partial(tf.layers.conv2d, filters=output_dim, padding='valid',
                    kernel_initializer=tf.variance_scaling_initializer, activation=None)
    # shortcut
    shortcut = upsample(net, output_dim)
    
    #
    output = net
    output = slim.batch_norm(output)
    output = tf.nn.relu(output)
    output = conv1(output)
    
    output = slim.batch_norm(output)
    output = tf.nn.relu(output)
    output = tf.pad(output, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), mode='REFLECT')
    output = conv2(output, kernel_size=kernel_size)
    
    return shortcut + output
    
    
def ResidualBlock(net, input_dim, output_dim, kernel_size):
    conv1 = partial(tf.layers.conv2d, filters=output_dim, padding='same',
                    kernel_initializer=tf.variance_scaling_initializer)
    conv2 = partial(tf.layers.conv2d, filters=output_dim, padding='same',
                    kernel_initializer=tf.variance_scaling_initializer)
                    
    if input_dim == output_dim:
        shortcut = net
    else:
        shortcut = conv1(net, kernel_size=kernel_size)
    
    #
    output = net
    output = slim.batch_norm(output)
    output = tf.nn.relu(output)
    output = conv1(output, kernel_size=kernel_size)
    
    output = slim.batch_norm(output)
    output = tf.nn.relu(output)
    output = conv2(output, kernel_size=kernel_size)
    
    return shortcut + output
    
