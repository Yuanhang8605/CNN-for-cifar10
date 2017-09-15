"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""

import numpy as np
import tensorflow as tf
import six
from net import Net
import ops

class ResNet(Net):
    """ ResNet model. """

    def __init__(self, cfgs, images, labels, mode='train', multi_gpu_mode=False):
        """ ResNet constructor. """
        Net.__init__(self, cfgs, images, labels, mode)
        self._use_bottleneck = cfgs['USE_BOTTLENECK'] 
        self._relu_leakiness = cfgs['RELU_LEAKINESS']
        self._num_residual_units = cfgs['NUM_RESIDUAL_UNITS']
        self._weight_decay_rate = cfgs['WEIGHT_DECAY_RATE']
        self.multi_gpu_mode = multi_gpu_mode
    
    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]   

    def inference(self):
        """ Build the core model within the gragh. 
        return:
            loggits before classifier
        """
        batch_size = self.images.get_shape()[0] 

        with tf.variable_scope('init'):
            x = self.images
            x = ops.conv('init_conv', x, 3, 3, 16, self._stride_arr(1), 'SAME', self.multi_gpu_mode)

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]

        if self._use_bottleneck:
            res_func = self._bottleneck_residual
            filters = [16, 64, 128, 258]
        else:
            res_func = self._residual
            filters = [16, 16, 32, 64]
        
        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]), 
                        activate_before_residual[0])
        for i in six.moves.range(1, self._num_residual_units):
            with tf.variable_scope('unit_1_%d'% i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)
            
        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                        activate_before_residual[1])
        for i in six.moves.range(1, self._num_residual_units):
            with tf.variable_scope('unit_2_%d'% i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                        activate_before_residual[2])
        for i in six.moves.range(1, self._num_residual_units):
            with tf.variable_scope('unit_3_%d'% i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)
        
        with tf.variable_scope('unit_last'):
            x, train_op = ops.batch_norm('final_bn', x, self.mode, 0.001, self.multi_gpu_mode)
            self.extra_train_ops.extend(train_op)
            x = ops.relu(x, self._relu_leakiness)
            x = ops.global_avg_pool(x)
        
        with tf.variable_scope('logit'):
            logits = ops.fc('fc1', x, batch_size, self.num_classes, self.multi_gpu_mode)
            #self.predictions = tf.nn.softmax(logits)
            self.logits = logits


    def loss(self):
        logits = self.logits
        with tf.variable_scope('loss'):
            ls = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels
            )
            ls = tf.reduce_mean(ls, name='ls')
            ls += ops.weight_decay(self._weight_decay_rate)
            # tf.summary.scalar('loss', ls)
            return ls


    def _bottleneck_residual(self, x, in_filter, out_filter, stride, 
                            activate_before_residual=False):
        """Bottleneck residual unit with 3 sub layers. """
        if activate_before_residual:
            with tf.variable_scope('common_bn_relu'):
                x, train_op = ops.batch_norm('init_bn', x, self.mode, 0.001, self.multi_gpu_mode)
                self.extra_train_ops.extend(train_op)
                x = ops.relu(x, self._relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu'):
                orig_x = x
                x, train_op = ops.batch_norm('init_bn', x, self.mode, 0.001, self.multi_gpu_mode)
                self.extra_train_ops.extend(train_op)
                x = ops.relu(x, self._relu_leakiness)
                
        with tf.variable_scope('sub1'):
            x = ops.conv('conv1', x, 1, in_filter, out_filter/4, stride, 'SAME', self.multi_gpu_mode) 
        
        with tf.variable_scope('sub2'):
            x, train_op = ops.batch_norm('bn2', x, self.mode, 0.001, self.multi_gpu_mode)
            self.extra_train_ops.extend(train_op)
            x = ops.relu(x, self._relu_leakiness)
            x = ops.conv('conv2', x, 3, out_filter/4, out_filter/4, [1,1,1,1], 'SAME', self.multi_gpu_mode)
        
        with tf.variable_scope('sub3'):
            x, train_op = ops.batch_norm('bn3', x, self.mode, 0.001, self.multi_gpu_mode)
            self.extra_train_ops.extend(train_op)
            x = ops.relu(x, self._relu_leakiness)
            x = ops.conv('conv3', x, 1, out_filter/4, out_filter, [1,1,1,1], 'SAME', self.multi_gpu_mode)

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = ops.conv('project', orig_x, 1, in_filter, out_filter, stride, 'SAME', self.multi_gpu_mode)
            x += orig_x
        tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _residual(self, x, in_filter, out_filter, stride, 
                    activate_before_residual=False):
        """ Residual unit with 2 sub layers. """
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x, train_op = ops.batch_norm('init_bn', x, self.mode, 0.001, self.multi_gpu_mode)
                self.extra_train_ops.extend(train_op)
                x = ops.relu(x, self._relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x, train_op = ops.batch_norm('init_bn', x, self.mode, 0.001, self.multi_gpu_mode)
                self.extra_train_ops.extend(train_op)
                x = ops.relu(x, self._relu_leakiness)
        
        with tf.variable_scope('sub1'):
            x = ops.conv('conv1' ,x, 3, in_filter, out_filter, stride, 'SAME', self.multi_gpu_mode)

        with tf.variable_scope('sub2'):
            x, train_op = ops.batch_norm('bn2', x, self.mode, 0.001, self.multi_gpu_mode)
            self.extra_train_ops.extend(train_op)
            x = ops.relu(x, self._relu_leakiness)
            x = ops.conv('conv2' ,x, 3, out_filter, out_filter, [1,1,1,1], 'SAME', self.multi_gpu_mode)
        
        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0,0], [0,0], [0,0],
                            [(out_filter-in_filter)//2, (out_filter-in_filter)//2]
                            ]
                )
            x += orig_x
        tf.logging.debug('image after unit %s', x.get_shape())
        return x
            
        
        