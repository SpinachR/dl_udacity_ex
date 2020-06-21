import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle


def save_list(data, path1, path2):
    with open(path1, 'wb+') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s..' % path1)
        
    with open(path2, 'w+') as file_handler:
        for item in data:
            file_handler.write("{}\n".format(item))
            
def plot_acc(accuracies, path):
    fig = plt.figure()
    for i, acc in enumerate(accuracies):
        plt.plot(np.arange(len(acc)), acc, 'C'+str(i+1), label=str(i+1))
        plt.legend()

    plt.savefig(path, bbox_inches='tight')
    plt.close()
        
        
def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter


def accuracy(predictions, labels):
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


# tf.nn.leaky_relu
def lrelu(x, leakiness=0.2):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def project_latent_vars(params, proj_shape, latent_vars, combine_method='sum'):
    '''
    Generate noise and project to input volume size.
    :param params:
    :param proj_shape: Shape to project noise (not including batch size).
    :param latent_vars: dictionary of `'key': Tensor of shape [batch_size, N]`
    :param combine_method:How to combine the projected values.
    :return:
        If combine_method=sum, a `Tensor` of size `hparams.projection_shape`
        If combine_method=concat and there are N latent vars, a `Tensor` of size
        `hparams.projection_shape`, with the last channel multiplied by N
    '''
    values = []
    for var in latent_vars:
        with tf.variable_scope(var):
            # Project & reshape noise to a HxWxC input
            projected = slim.fully_connected(
                latent_vars[var],
                int(np.prod(proj_shape)),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm)
            values.append(tf.reshape(projected, [params['batch_size']] + proj_shape))

    if combine_method == 'sum':
        result = values[0]
        for value in values[1:]:
            result += value
    elif combine_method == 'concat':
        # Concatenate along last axis
        result = tf.concat(values, len(proj_shape))
    else:
        raise ValueError('Unknown combine_method %s' % combine_method)

    tf.logging.info('Latent variables projected to size %s volume', result.shape)

    return result


def log_sum_exp(x, axis=1):
    '''
    :return: log[sum_i(exp(x_i))]
    '''
    m = tf.reduce_max(x, axis=axis)
    return m + tf.log(tf.reduce_sum(tf.exp(x - tf.expand_dims(m, 1)), axis=axis))
    
    
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    """Generate a generator that input a group of example in numpy.array and
    their labels, return the examples and labels by the given batchsize.

    Parameters
    ----------
    inputs : numpy.array
        (X) The input features, every row is a example.
    targets : numpy.array
        (y) The labels of inputs, every row is a example.
    batch_size : int
        The batch size.
    shuffle : boolean
        Indicating whether to use a shuffling queue, shuffle the dataset before return.

    Examples
    --------
    >>> X = np.asarray([['a','a'], ['b','b'], ['c','c'], ['d','d'], ['e','e'], ['f','f']])
    >>> y = np.asarray([0,1,2,3,4,5])
    >>> for batch in tl.iterate.minibatches(inputs=X, targets=y, batch_size=2, shuffle=False):
    >>>     print(batch)
    ... (array([['a', 'a'],
    ...        ['b', 'b']],
    ...         dtype='<U1'), array([0, 1]))
    ... (array([['c', 'c'],
    ...        ['d', 'd']],
    ...         dtype='<U1'), array([2, 3]))
    ... (array([['e', 'e'],
    ...        ['f', 'f']],
    ...         dtype='<U1'), array([4, 5]))


    Notes
    -------
    - If you have two inputs, e.g. X1 (1000, 100) and X2 (1000, 80), you can ``np.hstack((X1, X2))
    into (1000, 180) and feed into ``inputs``, then you can split a batch of X1 and X2.
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
