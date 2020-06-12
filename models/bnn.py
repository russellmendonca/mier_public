from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pickle
import os
import time
import itertools
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from models.utils import TensorStandardScaler
from models.fc import FC
from utils.logging import Progress, Silent
from parameterized_model import ParameterizedModel
from misc_utils import AttrDict

np.set_printoptions(precision=5)


class BNN:
    """Neural network models which model aleatoric uncertainty (and possibly epistemic uncertainty
    with ensembling).
    """

    def __init__(self, sess, obs_dim, act_dim, model_hyperparams):
        """Initializes a class instance.

        """
        self._sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim + act_dim
        self.output_dim = obs_dim + 1
        for key in model_hyperparams:
            setattr(self, key, model_hyperparams[key])
        # Instance variables

        self.decays, self.optvars, self.nonoptvars = [], [], []
        self.end_act, self.end_act_name = None, None
        self.scaler = None

        # Training objects
        self.sy_train_in, self.sy_train_targ = None, None
        self.train_op, self.mse_loss = None, None

        self.updated_parameters = []
        # Prediction objects
        self.sy_pred_in2d, self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac = None, None, None
        self.sy_pred_mean2d, self.sy_pred_var2d = None, None
        self.sy_pred_in3d, self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac = None, None, None

        self.build_graph()

    @property
    def is_probabilistic(self):
        return True

    @property
    def is_tf_model(self):
        return True

    @property
    def sess(self):
        return self._sess

    ###################################
    # Network Structure Setup Methods #
    ###################################

    def compile_output_layer(self, input_tensor):
        x = input_tensor
        #with tf.variable_scope(self.name):
        for i in range(4):
            # x = tf.layers.dense(x, 200, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, 200, activation=tf.nn.leaky_relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())

        x = tf.layers.dense(x, 2 * (self.output_dim), kernel_initializer=tf.contrib.layers.xavier_initializer())
        return x

    def build_graph(self):
        """Finalizes the network.
        """
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        #with tf.variable_scope(self.name):

        pm = ParameterizedModel(name=self.name +'_adaptable_parameters')
        with self.sess.as_default():
            with tf.variable_scope(self.name+'_non_adaptable_parameters'):
                self.scaler = TensorStandardScaler(self.obs_dim + self.act_dim)

                #if self.state_pred:
                self.max_state_logvar = tf.Variable(np.ones([self.obs_dim]) * 0.5,
                                                    dtype=tf.float32, name="max_state_log_var")
                self.min_state_logvar = tf.Variable(-np.ones([self.obs_dim]) * 10,
                                                    dtype=tf.float32, name="min_state_log_var")
               # if self.rew_pred:
                self.max_rew_logvar = tf.Variable(np.ones([1]) * 0.5,
                                                    dtype=tf.float32, name="max_rew_log_var")
                self.min_rew_logvar = tf.Variable(-np.ones([1]) * 10,
                                                    dtype=tf.float32, name="min_rew_log_var")

                # self.max_rew_logvar = tf.Variable(
                #     np.ones([1]) * (0.5 + np.log(1e-10 + self.reward_prediction_weight ** 2)),
                #     dtype=tf.float32,
                #     name="max_rew_log_var")
                # self.min_rew_logvar = tf.Variable(
                #     np.ones([1]) * (-10 + np.log(1e-10 + self.reward_prediction_weight ** 2)),
                #     dtype=tf.float32,
                #     name="min_rew_log_var")

        with pm.build_template():
            self.compile_output_layer(tf.zeros((256, self.input_dim)))
        # # import ipdb ; ipdb.set_trace()
        self.param = AttrDict()
        self.param.pm = pm
        self.param.parameter = pm.parameter

        self.param.trainables = [self.param.parameter]
        self.param.trainables_names = ['model_param']

        if not self.is_deterministic:
            #if self.state_pred:
            self.param.trainables.extend([self.max_state_logvar, self.min_state_logvar])
            self.param.trainables_names.extend(['max_state_logvar', 'min_state_logvar'])

            #if self.rew_pred:
            self.param.trainables.extend([self.max_rew_logvar, self.min_rew_logvar])
            self.param.trainables_names.extend(['max_rew_logvar', 'min_rew_logvar'])


        self.multi_task_placeholders = self.create_placeholders(num_tasks=self.meta_batch_size)
        self.single_task_placeholders = self.create_placeholders(num_tasks=1)
        #
        if self.meta_train:
            self.compile_meta_train()

        self.compile_single_task_test()
        self.compile_single_task_model_training()

        #### Setup prediction variables ######################
        with tf.variable_scope(self.name + '_prediction'):
            self.sy_pred_in2d = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.input_dim],
                                               name="2D_training_inputs")
            self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac = \
                self.create_prediction_tensors(self.sy_pred_in2d, factored=True)
            self.sy_pred_mean2d = tf.reduce_mean(self.sy_pred_mean2d_fac, axis=0)

            self.sy_pred_var2d = tf.reduce_mean(self.sy_pred_var2d_fac, axis=0) + \
                                 tf.reduce_mean(tf.square(self.sy_pred_mean2d_fac - self.sy_pred_mean2d), axis=0)

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

    def create_placeholders(self, num_tasks):
        # size is number of tasks X num nets X batch size X data_dim
        train_input = tf.placeholder(dtype=tf.float32,
                                     shape=[num_tasks, None, self.input_dim],
                                     name=self.name+"_train_input")
        train_target = tf.placeholder(dtype=tf.float32,
                                      shape=[num_tasks, None, self.output_dim],
                                      name=self.name+"_train_target")

        val_input = tf.placeholder(dtype=tf.float32,
                                   shape=[num_tasks, None, self.input_dim],
                                   name=self.name+"_val_input")
        val_target = tf.placeholder(dtype=tf.float32,
                                    shape=[num_tasks, None, self.output_dim],
                                    name=self.name+"_val_target")
        return train_input, train_target, val_input, val_target

    def meta_model_metric_keys(self, train_dicts=False):

        keys = ['mse_rew_loss', 'mse_state_loss', 'total_model_loss']

        if not self.is_deterministic:
            keys.extend(['total_rew_loss', 'total_state_loss'])
        if train_dicts:
            keys.extend(['grad_norm', 'param_norm'])

        return keys

    def compile_single_task_model_training(self):

        train_input, train_target, val_input, val_target = self.single_task_placeholders

        train_dict = self.get_parameterized_model_loss_dict(self.param.parameter, train_input[0], train_target[0])
        val_dict = self.get_parameterized_model_loss_dict(self.param.parameter, val_input[0], val_target[0])

        self.single_task_model_train_metrics = {'train_dicts': train_dict, 'val_dicts': val_dict}
        self.single_task_model_train_op = self.optimizer.minimize(train_dict['total_model_loss'],
                                                                  var_list=self.param.trainables)

    def compile_single_task_test(self):

        train_input, train_target, val_input, val_target = self.single_task_placeholders
        self.test_time_lr = tf.placeholder(dtype=tf.float32, shape=(), name='test_time_lr')

        self.model_prior_loss_dict = self.get_parameterized_model_loss_dict(self.param.parameter,
                                                                            train_input[0], train_target[0])
        grad = tf.clip_by_norm(tf.gradients(self.model_prior_loss_dict['total_model_loss'], self.param.parameter)[0],
                               self.clip_val_inner_grad)

        self.test_time_updated_param = self.param.parameter - (self.test_time_lr * grad)

    def compile_meta_train(self):
        # all_train_dicts = {}
        train_input, train_target, val_input, val_target = self.multi_task_placeholders
        post_adapt_loss = 0
        train_dicts_across_tasks = []
        val_dicts_across_tasks = []

        for task_id in range(self.meta_batch_size):
            updated_parameter, train_dicts, val_dicts = self.compile_per_task_adaptation_step \
                (train_input[task_id], train_target[task_id], val_input[task_id], val_target[task_id])

            # for i in range(1, self.fast_adapt_steps+1):
            post_adapt_loss += val_dicts[-1]['total_model_loss']

            train_dicts_across_tasks.append(train_dicts)
            val_dicts_across_tasks.append(val_dicts)

        self.meta_train_post_adapt_loss = post_adapt_loss / self.meta_batch_size

        self.meta_train_metrics = {'val_dicts': val_dicts_across_tasks,
                                   'train_dicts': train_dicts_across_tasks}

        self.setup_meta_gradient()

    def setup_meta_gradient(self):

        gvs = self.optimizer.compute_gradients(loss=self.meta_train_post_adapt_loss, var_list=self.param.trainables)
        gvs = [(g,v) for (g,v) in gvs if g!=None]
        _meta_update_norms = {}
        for i, gv in enumerate(gvs):
            _meta_update_norms[self.param.trainables_names[i] + '-gradient'] = tf.norm(gv[0])
            _meta_update_norms[self.param.trainables_names[i]] = tf.norm(gv[1])

        if self.clip_val_outer_grad:
            self.gvs = [(tf.clip_by_norm(grad, self.clip_val_outer_grad), var) for grad, var in gvs]
        else:
            self.gvs = gvs

        self.meta_train_metrics['meta_update_norms'] = _meta_update_norms
        self.metatrain_op = self.optimizer.apply_gradients(self.gvs)

    def compile_per_task_adaptation_step(self, train_input, train_target, val_input, val_target):
        updated_parameter = self.param.parameter
        train_dicts = []
        val_dicts = []
        for inner_step in range(self.fast_adapt_steps + 1):

            pre_adapt_dict = self.get_parameterized_model_loss_dict(updated_parameter, train_input, train_target)
            val_loss_dict = self.get_parameterized_model_loss_dict(updated_parameter, val_input, val_target)

            pre_adapt_dict['param_norm'] = tf.norm(updated_parameter)
            train_dicts.append(pre_adapt_dict)
            val_dicts.append(val_loss_dict)

            if inner_step < self.fast_adapt_steps:
                grad = tf.gradients(pre_adapt_dict['total_model_loss'], updated_parameter)[0]

                train_dicts[-1]['grad_norm'] = tf.norm(grad)
                clipped_grad = tf.clip_by_norm(grad, self.clip_val_inner_grad)
                updated_parameter = updated_parameter - (self.fast_adapt_lr * clipped_grad)

        return updated_parameter, train_dicts, val_dicts

    def predict(self, inputs, factored=False, *args, **kwargs):
        """Returns the distribution predicted by the model for each input vector in inputs.
        Behavior is affected by the dimensionality of inputs and factored as follows:

        inputs is 2D, factored=True: Each row is treated as an input vector.
            Returns a mean of shape [ensemble_size, batch_size, output_dim] and variance of shape
            [ensemble_size, batch_size, output_dim], where N(mean[i, j, :], diag([i, j, :])) is the
            predicted output distribution by the ith model in the ensemble on input vector j.

        inputs is 2D, factored=False: Each row is treated as an input vector.
            Returns a mean of shape [batch_size, output_dim] and variance of shape
            [batch_size, output_dim], where aggregation is performed as described in the paper.

        inputs is 3D, factored=True/False: Each row in the last dimension is treated as an input vector.
            Returns a mean of shape [ensemble_size, batch_size, output_dim] and variance of sha
            [ensemble_size, batch_size, output_dim], where N(mean[i, j, :], diag([i, j, :])) is the
            predicted output distribution by the ith model in the ensemble on input vector [i, j].

        Arguments:
            inputs (np.ndarray): An array of input vectors in rows. See above for behavior.
            factored (bool): See above for behavior.
        """
        if len(inputs.shape) == 2:
            if factored:
                return self.sess.run(
                    [self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac],
                    feed_dict={self.sy_pred_in2d: inputs}
                )
            else:
                return self.sess.run(
                    [self.sy_pred_mean2d, self.sy_pred_var2d],
                    feed_dict={self.sy_pred_in2d: inputs}
                )
        else:
            return self.sess.run(
                [self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac],
                feed_dict={self.sy_pred_in3d: inputs}
            )

    def create_prediction_tensors(self, inputs, factored=False, *args, **kwargs):
        """See predict() above for documentation.
        """
        with self.param.pm.build_parameterized(self.param.parameter):
            rew_mean, rew_logvar, state_mean, state_logvar = self.get_weighted_mean_logvar(
                self.compile_output_layer(inputs))

        factored_mean = tf.concat([rew_mean, state_mean], axis=-1)
        factored_variance = tf.math.exp(tf.concat([rew_logvar, state_logvar], axis=-1))

        if inputs.shape.ndims == 2 and not factored:
            mean = tf.reduce_mean(factored_mean, axis=0)
            variance = tf.reduce_mean(tf.square(factored_mean - mean), axis=0) + \
                       tf.reduce_mean(factored_variance, axis=0)
            return mean, variance
        return factored_mean, factored_variance

    def save_model(self, _dir, name, epoch):

        model_dir = os.path.join(_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        trainables = self.sess.run(self.param.trainables)
        pickle.dump(trainables, open(os.path.join(model_dir, 'epoch_' + str(epoch) + '.pkl'), 'wb'))

    def load_model(self, _file):

        loaded_vals = pickle.load(open(_file, 'rb'))
        for var, val in zip(self.param.trainables, loaded_vals):
            self.sess.run(tf.assign(var, val))

    def get_parameterized_model_loss_dict(self, param, _input, _target):

        with self.param.pm.build_parameterized(param):
            output_layer = self.compile_output_layer(_input)

        return self._get_model_loss_dict(output_layer, _target)

    def _get_model_loss_dict(self, output_layer, targets):

        total_model_loss = 0
        loss_dict = self._compile_losses(output_layer, targets)
        loss_key = 'mse' if self.is_deterministic else 'total'
        if self.rew_pred:
            total_model_loss += loss_dict[loss_key + '_rew_loss']
        if self.state_pred:
            total_model_loss += loss_dict[loss_key + '_state_loss']
        loss_dict['total_model_loss'] = total_model_loss
        return loss_dict

    def _compile_losses(self, output_layer, targets):

        rew_mean, rew_logvar, state_mean, state_logvar = self.get_weighted_mean_logvar(output_layer)

        total_rew_loss, mse_rew_loss = self._compile_log_likelihood(rew_mean, rew_logvar, targets[:, :1],
                                                                    self.min_rew_logvar, self.max_rew_logvar)
        total_state_loss, mse_state_loss = self._compile_log_likelihood(state_mean, state_logvar, targets[:, 1:],
                                                                        self.min_state_logvar, self.max_state_logvar)

        return OrderedDict({'total_rew_loss': total_rew_loss,
                            'total_state_loss': total_state_loss,
                            'mse_rew_loss': mse_rew_loss,
                            'mse_state_loss': mse_state_loss})

    def get_weighted_mean_logvar(self, cur_out):

        mean, logvar = cur_out[:, :self.output_dim], cur_out[:, self.output_dim:]

        # rew_mean = mean[:, :1] * self.reward_prediction_weight
        # rew_logvar = self._incorporate_max_min_logvar(logvar[:, :1] + np.log(self.reward_prediction_weight ** 2),
        #                                               self.max_rew_logvar, self.min_rew_logvar)
        rew_mean = mean[:, :1]
        rew_logvar = self._incorporate_max_min_logvar(logvar[:, :1] , self.max_rew_logvar, self.min_rew_logvar)
        state_mean = mean[:, 1:]
        state_logvar = self._incorporate_max_min_logvar(logvar[:, 1:], self.max_state_logvar, self.min_state_logvar)

        return rew_mean, rew_logvar, state_mean, state_logvar

    def _incorporate_max_min_logvar(self, logvar, max_logvar, min_logvar):
        logvar = max_logvar - tf.nn.softplus(max_logvar - logvar)
        return min_logvar + tf.nn.softplus(logvar - min_logvar)

    def _compile_log_likelihood(self, mean, log_var, targets, min_logvar, max_logvar):
        """Helper method for compiling the loss function.

        The loss function is obtained from the log likelihood, assuming that the output
        distribution is Gaussian, with both mean and (diagonal) covariance matrix being determined
        by network outputs.

        Arguments:
            inputs: (tf.Tensor) A tensor representing the input batch
            targets: (tf.Tensor) The desired targets for each input vector in inputs.
            inc_var_loss: (bool) If True, includes log variance loss.

        Returns: (tf.Tensor) A tensor representing the loss on the input arguments.
        """
        inv_var = tf.exp(-log_var)
        mse_losses = tf.reduce_mean(tf.square(mean - targets))
        # if inc_var_loss:
        mean_losses = tf.reduce_mean(tf.reduce_mean(tf.square(mean - targets) * inv_var, axis=-1), axis=-1)
        var_losses = tf.reduce_mean(tf.reduce_mean(log_var, axis=-1), axis=-1)

        total_losses = mean_losses + var_losses + \
                       0.01 * tf.reduce_sum(max_logvar) - 0.01 * tf.reduce_sum(min_logvar)

        return total_losses, mse_losses
