"""
Base class for cnn net, define common variable and interface 
"""

import numpy as np
import tensorflow as tf

class Net(object):
    """ base class for cnn net. """
    def __init__(self, cfgs, images, labels, mode): 
        self.num_classes = cfgs['NUM_CLASSES']
        self.mode = mode     # mode = train or test

        """ Batch size set up """
        #self.train_batch_size = cfgs['TRAIN_BATCH_SIZE']
        #self.eval_batch_size = cfgs['EVAL_BATCH_SIZE']
        #self.test_batch_size = cfgs['TEST_BATCH_SIZE']
        
        """ interface to images and labels """
        #self.images = tf.placeholder(dtype=tf.float32, shape=(None, self.im_size[0], self.im_size[1], self.im_channels))
        #self.labels = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.images = images
        self.labels = labels
        self.extra_train_ops = []

    def inference(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError
    




