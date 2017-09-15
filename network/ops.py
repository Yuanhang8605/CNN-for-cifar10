import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages

"""
This file capsulate common layer ops in CNN
"""

# use_multi_gpus = False

def weight_decay(weight_decay_rate):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find(r'DW') > 0:
            costs.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)

    return tf.multiply(weight_decay_rate, tf.add_n(costs))


def conv(name, x, filter_size, in_filters, out_filters, strides, padmod, multi_gpu_mode=False):
    """Convolution op."""
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        if multi_gpu_mode:
            with tf.device('/cpu:0'):
                kernel = tf.get_variable(
                    'DW', [filter_size, filter_size, in_filters, out_filters],
                    tf.float32, initializer=tf.random_normal_initializer(
                        stddev=np.sqrt(2.0/n)))
        else:
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0/n)))
        return tf.nn.conv2d(x, kernel, strides, padding=padmod)


def relu(x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def fc(name, x, batch_size, out_dim, multi_gpu_mode=False):
    """FullyConnected layer. """
    with tf.variable_scope(name):
        if x.get_shape()[0] != batch_size:
            x = tf.reshape(x, [batch_size, -1])
        if multi_gpu_mode:
            with tf.device('/cpu:0'):
                w = tf.get_variable(
                    'DW', [x.get_shape()[1], out_dim],
                    initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
                b = tf.get_variable('biases', [out_dim],
                                    initializer=tf.constant_initializer())
        else:
            w = tf.get_variable(
                'DW', [x.get_shape()[1], out_dim],
                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            b = tf.get_variable('biases', [out_dim],
                                initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)


def global_avg_pool(x):
    """global avarage pooling op"""
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


def batch_norm(name, x, mode, epsilon=0.001, multi_gpu_mode=False):
    """Batch normalization."""
    train_ops = []
    ndims = x.get_shape().ndims  

    with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]
        if multi_gpu_mode: 
            with tf.device('/cpu:0'):
                beta = tf.get_variable(
                    'beta', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32))
                gamma = tf.get_variable(
                    'gamma', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32))
        else:
            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

        if mode == 'train':
            if ndims == 4:
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            elif ndims == 2:
                mean, variance = tf.nn.moments(x, [0], name='moments')
            else:
                raise ValueError("The input tensor has wrong ndims!")
            if multi_gpu_mode:
                with tf.device('/cpu:0'):
                    moving_mean = tf.get_variable(
                        'moving_mean', params_shape, tf.float32,
                        initializer=tf.constant_initializer(0.0, tf.float32),
                        trainable=False)
                    moving_variance = tf.get_variable(
                        'moving_variance', params_shape, tf.float32,
                        initializer=tf.constant_initializer(1.0, tf.float32),
                        trainable=False)
            else:
                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

            train_ops.append(moving_averages.assign_moving_average(
                moving_mean, mean, 0.9))
            train_ops.append(moving_averages.assign_moving_average(
                moving_variance, variance, 0.9))
        else:
            if multi_gpu_mode:
                with tf.device('/cpu:0'):
                    mean = tf.get_variable(
                        'moving_mean', params_shape, tf.float32,
                        initializer=tf.constant_initializer(0.0, tf.float32),
                        trainable=False)
                    variance = tf.get_variable(
                        'moving_variance', params_shape, tf.float32,
                        initializer=tf.constant_initializer(1.0, tf.float32),
                        trainable=False)
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

            tf.summary.histogram(mean.op.name, mean)
            tf.summary.histogram(variance.op.name, variance)
        # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, epsilon)
        y.set_shape(x.get_shape())
        return (y, train_ops)


def one_hot_encoding(labels, num_classes, scope=None):
    """Transform numeric labels into onehot_labels.

    Args:
        labels: [batch_size] target labels.
        num_classes: total number of classes.
        scope: Optional scope for name_scope.
    Returns:
        one hot encoding of the labels.
    """
    with tf.name_scope(scope, 'OneHotEncoding', [labels]):
        batch_size = labels.get_shape()[0]
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        labels = tf.cast(tf.expand_dims(labels, 1), indices.dtype)
        concated = tf.concat(axis=1, values=[indices, labels])
        onehot_labels = tf.sparse_to_dense(
            concated, tf.stack([batch_size, num_classes]), 1.0, 0.0)
        onehot_labels.set_shape([batch_size, num_classes])
        return onehot_labels





