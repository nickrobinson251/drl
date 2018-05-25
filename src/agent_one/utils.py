from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import math
import numpy as np

def translate(value, old_range, new_range):
    OldRange = float(old_range[1] - old_range[0])
    NewRange = float(new_range[1] - new_range[0])
    NewValue = (value - old_range[0]) * NewRange / float(OldRange) + new_range[0]
    return NewValue

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis = 0)[::-1]

def log_likelihood_tf(x, means, logstds):
    zs = (x - means)/tf.exp(logstds)
    return -tf.reduce_sum(logstds, -1) - .5 *tf.reduce_sum(tf.square(zs), -1) - .5*means.get_shape()[-1].value * np.log(2*np.pi)

def log_likelihood(x, means, logstds):
    if x.ndim < 2:  # no batch
        x = x[np.newaxis, :]
        means = means[np.newaxis, :]
        logstds = logstds[np.newaxis, :]

    zs = (x - means)/np.exp(logstds)
    return np.nansum(logstds, axis=1) - 0.5*np.nansum(np.square(zs), axis=1) - 0.5*np.shape(means)[1]*np.log(2*np.pi)
    # log_prob = -0.5*np.square(zs)-0.5*np.log(2*np.pi)-logstds
    # return np.nansum(log_prob, axis=1)

def flatten_var(var_list):
    return tf.concat([tf.reshape(var, [tf.size(var)]) for var in var_list], axis = 0)

def set_from_flat(var_list, x):
    start = 0
    assigns = []
    for var in var_list:
        shape = var.get_shape().as_list()
        size = np.prod(shape)
        assigns.append(tf.assign(var, tf.reshape(x[start:start + size], shape)))
        start += size
    return assigns

def kl_sym(mean_1, logstd_1, mean_2, logstd_2):
    std_1 = tf.exp(logstd_1)
    std_2 = tf.exp(logstd_2)
    numerator = tf.square(mean_1 - mean_2) + tf.square(std_1) - tf.square(std_2)
    denominator = 2 * tf.square(std_2) + 1e-8
    kl = tf.reduce_sum(numerator/denominator + logstd_2 - logstd_1, -1)
    return kl

def kl_sym_firstfixed(mean, logstd):
    m_1, ls_1 = map(tf.stop_gradient, [mean, logstd])
    m_2, ls_2 = mean, logstd
    return kl_sym(m_1, ls_1, m_2, ls_2)

def entropy(logstds):
	return tf.reduce_sum(logstds + .5 * np.log(2.*np.pi*np.e), axis = -1)