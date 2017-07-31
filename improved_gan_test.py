import numpy as np
import theano.tensor as T


def log_sum_exp_1(x, axis=1):
    return T.log(T.sum(T.exp(x), axis))


def log_sum_exp(x, axis=1):
    m = T.max(x, axis)
    return m+T.log(T.sum(T.exp(x-m.dimshuffle(0, 'x')), axis=axis))

labels = T.arange(3, dtype='int64')

logit = np.array([[9, 1, 3, 0, 2], [1, 9, 2, 0, 3], [1, 2, 9, 0, 1]])

z_exp = T.mean(log_sum_exp(logit))


l_lab = logit[[0, 1, 2], [0, 1, 2]]   # x's logit corresponds to its real label

'''
This is really a cross-entropy
    -T.mean(l_lab): corresponding logit of x
    T.mean(z_exp): log(sum[exp(logit_1) + exp(logit_2) + ... + exp(logit_k)])
    cross-entropy = -{ log(exp(logit)) - log(sum_j[exp(logit_j]) }
'''
loss_lab = -T.mean(l_lab) + T.mean(z_exp)
print(loss_lab.eval())

logit_unl = np.array([[9, 1, 0, 0, 3], [0, 9, 2, 0, 5], [2, 1, 9, 0, 3]])
logit_fake = np.array([[9, 1, 0, -2, 3], [0, 9, 2, -1, 5], [2, 1, 9, 0, 3]])

l_unl = log_sum_exp(logit_unl)
l_unl_1 = T.nnet.softplus(log_sum_exp(logit_unl))
l_fake_2 = T.nnet.softplus(log_sum_exp(logit_fake))

print('l_unl: ', l_unl.eval())
print('l_unl_1: ', l_unl_1.eval())
print('l_fake_2: ', l_fake_2.eval())

'''
    log_sum_exp = sum_j(exp(logit_j))
    loss_unl = - { log([log_sum_exp]/[1+log_sum_exp]) + log([1]/[1+log_sum_exp])}
'''
loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(l_unl_1) + 0.5 * T.mean(l_fake_2)
print(loss_unl.eval())





