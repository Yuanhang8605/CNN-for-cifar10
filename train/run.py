import tensorflow as tf
import numpy as np

import time
import six
import sys

sys.path.append('..')

#from network.resnet import ResNet
from network.vggnet import VggNet
from solver import Solver 
from data import cifar_input 
#from config.resnet_config import CONFIG
from config.vggnet_config import CONFIG

#### Get command line args
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_bool('evaluate_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_bool('lr_setHook', False,
                         'Whether setHook the learning rate in the training process.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')

def main(_):
    '''
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus > 0:
        dev = '/gpu:0'
    # else:
        # raise NotImplementedError("Only support 0 or 1 gpu now. ")
    '''
    
    if FLAGS.mode == 'train':
        batch_size = CONFIG['TRAIN_BATCH_SIZE']
        data_path =  FLAGS.train_data_path
    elif FLAGS.mode == 'eval':
        batch_size = CONFIG['EVAL_BATCH_SIZE']
        data_path = FLAGS.eval_data_path
    
    # num_classes = CONFIG['NUM_CLASSES']
    multi_gpu_mode = False
    if FLAGS.mode == 'train':
        if FLAGS.num_gpus > 1:
            multi_gpu_mode = True


    # get tf queued images
    images, labels = cifar_input.build_input(
        FLAGS.dataset, data_path, batch_size, FLAGS.mode
    )

    # Construct a ResNet
    net = VggNet(CONFIG, images, labels, FLAGS.mode, multi_gpu_mode)

    # Construct a Solver
    solver = Solver(net, CONFIG, FLAGS.mode, FLAGS.lr_setHook, FLAGS.num_gpus)

    if FLAGS.mode == 'train':
        solver.train()
    elif FLAGS.mode == 'eval':
        solver.evaluate(FLAGS.eval_batch_count, FLAGS.evaluate_once)

    # train or eval
    '''
    if FLAGS.num_gpus < 2:
        with tf.device(dev):
            if FLAGS.mode == 'train':
                solver.train()
            elif FLAGS.mode == 'eval':
                solver.evaluate(FLAGS.eval_batch_count, FLAGS.evaluate_once)
    else:
        if FLAGS.mode == 'train':
            solver.train_multi_GPU(FLAGS.num_gpus)
        elif FLAGS.mode == 'eval':
            with tf.device(dev):
                solver.evaluate(FLAGS.eval_batch_count, FLAGS.evaluate_once)
    '''

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
