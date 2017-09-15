""" Train class to config training
"""

import tensorflow as tf
import numpy as np
import sys 
import six
import time

class Solver(object):
    """ Solver class to config train process """
    def __init__(self, model, cfgs, mode='train', lr_setHook=False, num_gpus=0):
        """ Config the parameters """
        self.model = model
        self.mode = mode
        self._lrn_rate = cfgs['LRN_RATE']
        # self.lrn_decay = cfgs[]
        # self.min_lrn_rate=cfgs['MIN_LRN_RATE']
        self.optimizer = cfgs['OPTIMIZER']
        # self.train_batch_size = cfgs['TRAIN_B)ATCH_SIZE']
        # self.eval_batch_size = cfgs['EVAL_BATCH_SIZE']
        self._checkpoint_dir = cfgs['CHECKPOINT_DIR']
        self._log_dir = cfgs['LOG_DIR']
        self._eval_dir = cfgs['EVAL_DIR']
        self._max_iter = cfgs['MAX_ITER']
        # self._eval_batch_size = cfgs['EVAL_BATCH_SIZE']
        # self._train_batch_size = cfgs['TRAIN_BATCH_SIZE']
        self.lr_setHook = lr_setHook
        self.num_gpus = num_gpus
    

    def _reset(self):
        pass
    

    def _build_graph_single_dev(self):
        """ Build gragh for single device """
        if self.num_gpus == 0:
            dev = '/cpu:0'
        elif self.num_gpus > 0:
            dev = '/gpu:0'

        with tf.device(dev):
            self.global_step = tf.contrib.framework.get_or_create_global_step()

            self.lrn_rate = tf.constant(self._lrn_rate, tf.float32)
            tf.summary.scalar('learning_rate', self.lrn_rate)

            self.model.inference()
            self.logits = self.model.logits
            self.labels = self.model.labels
            # self._predictions = tf.nn.softmax(self.model.logits)
            self.loss = self.model.loss()
            tf.summary.scalar('loss', self.loss)

            if self.mode == 'train':
                self._build_train_op_single_dev()
            
            self.summaries = tf.summary.merge_all()


    def _build_train_op_single_dev(self):
        """ Build training specific ops for the graph. """
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_variables)

        if self.optimizer == 'sgd':
            optim = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.optimizer == 'mom':
            optim = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        elif self.optimizer == 'adam':
            optim = tf.train.AdamOptimizer(self.lrn_rate)
        elif self.optimizer == 'adagrad':
            optim = tf.train.AdagradOptimizer(self.lrn_rate)
        
        apply_op = optim.apply_gradients(
            zip(grads, trainable_variables), 
            global_step=self.global_step, name='train_step'
        )
        train_ops = [apply_op] + self.model.extra_train_ops
        self.train_op = tf.group(*train_ops)


    def train(self):
        """ Construct SGD training loop. """
        if self.num_gpus < 2:
            self._build_graph_single_dev()
        else:
            self._build_graph_multi_gpu()

        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.
                TRAINABLE_VARS_PARAMS_STAT_OPTIONS
        )

        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
        
        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)    
        
        truth = tf.argmax(self.labels, axis=1)

        predictions = tf.nn.softmax(self.logits)
        predictions = tf.argmax(predictions, axis=1)
        precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

        summary_hook = tf.train.SummarySaverHook(
            save_steps = 100,
            output_dir = self._log_dir,
            summary_op=tf.summary.merge([self.summaries,
                    tf.summary.scalar('Precision', precision)])
        )

        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': self.global_step, 
                     'loss': self.loss, 
                     'precision': precision},
            every_n_iter=100
        )

        # define SessionRunHook to control the lrn_rate during training process
        global_step_ = self.global_step
        lrn_rate_ = self.lrn_rate
        init_lrn_rate = self._lrn_rate

        class _LearningRateSetterHook(tf.train.SessionRunHook):
            """ Sets learning_rate based on global step. """

            def begin(self):
                self._lr = init_lrn_rate
            
            def before_run(self, run_context):
                return tf.train.SessionRunArgs(
                   global_step_, # Asks for global step value. 
                   feed_dict={lrn_rate_: self._lr} # Sets learning rate
                )
            
            def after_run(self, run_context, run_values):
                train_step = run_values.results
                if train_step < 40000:
                    #self._lrn_rate = 0.1
                    pass
                elif train_step < 60000:
                    self._lr /= 10.0
                elif train_step < 80000:
                    self._lr /= 10.0
                else:
                    self._lr /= 10.0
        
        if self.lr_setHook:
            hooks_ = [logging_hook, _LearningRateSetterHook()]
        else:
            hooks_ = [logging_hook]


        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=self._checkpoint_dir,
            hooks=hooks_,
            chief_only_hooks=[summary_hook],
            # Since we provide a SummarySaverHook, we need to disable default
            # SummarySaverHook. To do that we set save_summaries_steps to 0.            
            save_summaries_steps=0,
            config=tf.ConfigProto(allow_soft_placement=True)
            ) as mon_sess:

            iter_num = 0
            while not mon_sess.should_stop(): 
                mon_sess.run(self.train_op)
                if iter_num > self._max_iter: 
                    break
                iter_num += 1
    

    def _build_graph_multi_gpu(self):
        with tf.device('/cpu:0'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.lrn_rate = tf.constant(self._lrn_rate, tf.float32)
            tf.summary.scalar('learning_rate', self.lrn_rate)

            # Create an optimizer that perform gradient descent. 
            if self.optimizer == 'sgd':
                optim = tf.train.GradientDescentOptimizer(self.lrn_rate)
            elif self.optimizer == 'mom':
                optim = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
            elif self.optimizer == 'adam':
                optim = tf.train.AdamOptimizer(self.lrn_rate)
            elif self.optimizer == 'adagrad':
                optim = tf.train.AdagradOptimizer(self.lrn_rate)
 
            tower_grads = []
            tower_losses = []
            tower_logits = []
            tower_labels = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in six.moves.range(self.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_%d' % i) as scope:
                            self.model.inference()
                            loss = self.model.loss()
                            tower_labels.append(self.model.labels)
                            tower_losses.append(loss)
                            tf.get_variable_scope().reuse_variables()
                            tower_logits.append(self.model.logits)
                            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                            # Calculate the gradients for the batch of data in this tower
                            grads = optim.compute_gradients(loss)
                            tower_grads.append(grads)

            # We must calculate the mean of each gradient
            self.logits = tf.concat(tower_logits, axis=0, name='tower_concat')
            self.labels = tf.concat(tower_labels, axis=0, name='tower_labels')
            grads = self._average_gradients(tower_grads)
            self.loss = tf.add_n(tower_losses, 'tower_losses')
            tf.summary.scalar('loss', self.loss)

            for grad, var in grads:
                if grad is not None:
                    self.summaries.append(tf.summary.histogram(var.op.name, var))
            
            apply_op = optim.apply_gradients(grads, global_step=self.global_step)
            train_ops = [apply_op] + self.model.extra_train_ops
            self.train_op = tf.group(*train_ops)

            self.summaries = tf.summary.merge_all()
 

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.   

            # def tower_loss(scope):    """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


    def evaluate(self, eval_batch_count, evaluate_once='False'):
        """ Eval loop. """ 
        self._build_graph_single_dev()
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(self._eval_dir)

        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        
        tf.train.start_queue_runners(sess)
        best_precision = 0.0

        try:
            ckpt_state = tf.train.get_checkpoint_state(self._checkpoint_dir)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoints: %s', e)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', self._checkpoint_dir)

        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        while True:
            '''
            try:
                ckpt_state = tf.train.get_checkpoint_state(self._checkpoint_dir)
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoints: %s', e)
                continue
            if not (ckpt_state and ckpt_state.model_checkpoint_path):
                tf.logging.info('No model to eval yet at %s', self._checkpoint_dir)
                continue
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            '''

            total_prediction, correct_prediction = 0, 0
            predictions_ = tf.nn.softmax(self.logits)
            for _ in six.moves.range(eval_batch_count):
                (summaries, loss, predictions, truth, train_step) = sess.run(
                [self.summaries, self.loss, predictions_,
                self.model.labels, self.global_step])

                truth = np.argmax(truth, axis=1)
                predictions = np.argmax(predictions, axis=1)
                correct_prediction += np.sum(truth == predictions)
                total_prediction += predictions.shape[0]

            precision = 1.0 * correct_prediction / total_prediction
            best_precision = max(precision, best_precision)

            precision_summ = tf.Summary()
            precision_summ.value.add(
                tag='Precision', simple_value=precision)
            summary_writer.add_summary(precision_summ, train_step)
            best_precision_summ = tf.Summary()
            best_precision_summ.value.add(
                tag='Best Precision', simple_value=best_precision)
            summary_writer.add_summary(best_precision_summ, train_step)
            summary_writer.add_summary(summaries, train_step)
            tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                            (loss, precision, best_precision))
            summary_writer.flush()

            if evaluate_once:
                break

            # time.sleep(60)



