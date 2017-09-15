"""VGGnet model.
"""

import numpy as np
import tensorflow as tf
import six
from net import Net
import ops

class VggNet(Net):
    """ VggNet model. """

    def __init__(self, cfgs, images, labels, mode='train', multi_gpu_mode=False):
        """ ResNet constructor. """
        Net.__init__(self, cfgs, images, labels, mode)
        self._relu_leakiness = cfgs['RELU_LEAKINESS']
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
            x = ops.conv('init_conv1', x, 3, 3, 16, self._stride_arr(1), 'SAME', self.multi_gpu_mode)
            x, train_op = ops.batch_norm('bn1', x, self.mode, 0.001, self.multi_gpu_mode)
            self.extra_train_ops.extend(train_op)
            x = ops.relu(x, self._relu_leakiness)

            x = ops.conv('init_conv2', x ,3, 16, 16, self._stride_arr(1), 'SAME', self.multi_gpu_mode)
            x, train_op = ops.batch_norm('bn2', x, self.mode, 0.001, self.multi_gpu_mode)
            self.extra_train_ops.extend(train_op)
            x = ops.relu(x, self._relu_leakiness)
            
        with tf.variable_scope('unit_1_0'):
            x = tf.nn.max_pool(x, [1,2,2,1] ,self._stride_arr(2), 'VALID', name='max_pool')
            x = ops.conv('conv', x, 3, 16, 32, self._stride_arr(1), 'SAME', self.multi_gpu_mode)
            x, train_op = ops.batch_norm('bn', x, self.mode, 0.001, self.multi_gpu_mode)
            self.extra_train_ops.extend(train_op)
            x = ops.relu(x, self._relu_leakiness)
        for i in six.moves.range(1, 2):
            with tf.variable_scope('unit_1_%d' % i):
                x = ops.conv('conv', x, 3, 32, 32, self._stride_arr(1), 'SAME', self.multi_gpu_mode)
                x, train_op = ops.batch_norm('bn', x, self.mode, 0.001, self.multi_gpu_mode)
                self.extra_train_ops.extend(train_op)
                x = ops.relu(x, self._relu_leakiness)
       
        with tf.variable_scope('unit_2_0'):
            x = tf.nn.max_pool(x, [1,2,2,1] ,self._stride_arr(2), 'VALID', name='max_pool')
            x = ops.conv('conv', x, 3, 32, 64, self._stride_arr(1), 'SAME', self.multi_gpu_mode)
            x, train_op = ops.batch_norm('bn', x, self.mode, 0.001, self.multi_gpu_mode)
            self.extra_train_ops.extend(train_op)
            x = ops.relu(x, self._relu_leakiness)
        for i in six.moves.range(1, 4):
            with tf.variable_scope('unit_2_%d' % i):
                x = ops.conv('conv', x, 3, 64, 64, self._stride_arr(1), 'SAME', self.multi_gpu_mode)
                x, train_op = ops.batch_norm('bn', x, self.mode, 0.001, self.multi_gpu_mode)
                self.extra_train_ops.extend(train_op)
                x = ops.relu(x, self._relu_leakiness)
        
        with tf.variable_scope('unit_3_0'):
            x = tf.nn.max_pool(x, [1,2,2,1] ,self._stride_arr(2), 'VALID', name='max_pool')
            x = ops.conv('conv', x, 3, 64, 128, self._stride_arr(1), 'SAME', self.multi_gpu_mode)
            x, train_op = ops.batch_norm('bn', x, self.mode, 0.001, self.multi_gpu_mode)
            self.extra_train_ops.extend(train_op)
            x = ops.relu(x, self._relu_leakiness)
        for i in six.moves.range(1, 4):
            with tf.variable_scope('unit_3_%d' % i):
                x = ops.conv('conv', x, 3, 128, 128, self._stride_arr(1), 'SAME', self.multi_gpu_mode)
                x, train_op = ops.batch_norm('bn', x, self.mode, 0.001, self.multi_gpu_mode)
                self.extra_train_ops.extend(train_op)
                x = ops.relu(x, self._relu_leakiness)
       
        with tf.variable_scope('unit_last'):
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
