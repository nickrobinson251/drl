from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal
import math
import numpy as np

# def translate(value, old_range, new_range):
#     OldRange = float(old_range[1] - old_range[0])
#     NewRange = float(new_range[1] - new_range[0])
#     NewValue = (value - old_range[0]) * NewRange / float(OldRange) + new_range[0]
#     return NewValue

def translate(value, old_range, new_range):
    value = np.array(value)
    old_range = np.array(old_range)
    new_range = np.array(new_range)

    OldRange = float(old_range[1][:] - old_range[0][:])
    NewRange = float(new_range[1][:] - new_range[0][:])
    NewValue = (value - old_range[0][:]) * NewRange / float(OldRange) + new_range[0][:]
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

def linesearch(f, x, fullstep, max_backtracks, max_kl):
    fval, kl = f(x)
    for (_n_backtracks, step_frac) in enumerate(.5 ** np.arange(max_backtracks)):
        new_x = x + step_frac*fullstep
        newfval, newkl = f(new_x)
        if newfval <= fval and newkl <= max_kl:
            # print('valid gradient')
            return new_x
    return x

def compute_hessian(fn, vars):
    mat = []
    for v1 in vars:
        temp = []
        for v2 in vars:
            # computing derivative twice, first w.r.t v2 and then w.r.t v1
            temp.append(tf.gradients(tf.gradients(fn, v2)[0], v1)[0])
        temp = [tf.constant(0) if t == None else t for t in temp] # tensorflow returns None when there is no gradient, so we replace None with 0
        temp = tf.stack(temp) # tf.pack replaced by stack
        mat.append(temp)
    mat = tf.stack(mat)
    return mat


def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / (p.dot(z) + 1e-8)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / (rdotr + 1e-8)
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x

def preconditioned_cg(f_Ax, f_Minvx, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 318
    """
    x = np.zeros_like(b)
    r = b.copy()
    p = f_Minvx(b)
    y = p
    ydotr = y.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x, f_Ax)
        if verbose: print(fmtstr % (i, ydotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = ydotr / p.dot(z)
        x += v * p
        r -= v * z
        y = f_Minvx(r)
        newydotr = y.dot(r)
        mu = newydotr / ydotr
        p = y + mu * p

        ydotr = newydotr

        if ydotr < residual_tol:
            break

    if verbose: print(fmtstr % (cg_iters, ydotr, np.linalg.norm(x)))

    return x

# y -> Hy
def hessian_vector_product(func, vrbs, y):
    first_derivative = tf.gradients(func, vrbs)
    flat_y = list()
    start = 0
    for var in vrbs:
        variable_size = np.prod(var.get_shape().as_list())
        param = tf.reshape(y[start:(start + variable_size)], var.get_shape())
        flat_y.append(param)
        start += variable_size
    # First derivative * y
    gradient_with_y = [tf.reduce_sum(f_d * f_y) for (f_d, f_y) in zip(first_derivative, flat_y)]
    grad = tf.gradients(gradient_with_y, vrbs)
    HVP = flatten_var(grad)
    #HVP = FLAT_GRAD(gradient_with_y, vrbs)
    return HVP

def entropy(logstds):
	return tf.reduce_sum(logstds + .5 * np.log(2.*np.pi*np.e), axis = -1)
