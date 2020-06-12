import tensorflow as tf
import numpy as np
import os
import pickle
from setup import setup
from data import TrajData, prepare_data, prepare_multitask_data, append_context_to_array, append_context_to_traj, \
    concatenate_traj_data, dict_diff, split_data_into_train_val, sample_from_buffer_and_append_context

from misc_utils import avg_metrics_across_tasks, discard_term_states, print_model_losses

class MIER:

    def __init__(self, variant):

        for key in variant:
            setattr(self, key, variant[key])
        setup(self)

    ################################## MIER  Training #####################################################
    def train(self):

        if self.pre_adapt_replay_buffer_load_path == None:
            print('initial exploration')
            for task_id in range(self.n_train_tasks):
                print('task', task_id)
                for _ in range(self.initial_exploration_repeats):
                    self.collect_data_for_metatraining(task_id, collect_with_updated_context=False)

        for epoch in range(self.num_train_epochs):

            tasks = np.random.choice(np.arange(self.n_train_tasks), self.num_tasks_sample)
            for task_id in tasks:
                self.collect_data_for_metatraining(task_id, collect_with_updated_context=True)

            # saving models
            if epoch % self.save_interval == 0:
                self.save_models_and_buffers(epoch)

            # tasks for training this epoch
            tasks = np.random.choice(np.arange(self.n_train_tasks), self.model.meta_batch_size,
                                     replace=self.model.meta_batch_size > self.n_train_tasks)
            # meta-train the model
            self.run_training_epoch(tasks, epoch)
            self.eval_during_training(epoch)

    def collect_data_for_metatraining(self, task_id, collect_with_updated_context=True):

        self.sampler.reset_task(task_id)
        data = self.sampler.sample(self.num_sample_steps_prior, self.model.get_context())
        self.pre_adapt_replay_buffer.add(data, task_id)
        self.replay_buffer.add(data, task_id)

        if collect_with_updated_context:
            proc_data = prepare_data(data)
            updated_context = self.model.get_updated_context(proc_data[0], proc_data[1])
            post_update_data = self.sampler.sample(self.num_sample_steps_updated_context, updated_context)
            self.replay_buffer.add(post_update_data, task_id)

    def run_training_epoch(self, tasks, epoch):

        for step in range(self.num_training_steps_per_epoch):

            if self.joint_state_reward_model:
                self.run_meta_training_step(self.model, tasks, epoch, step, mode='joint')

            else:
                self.meta_training_step(self.reward_model, tasks, epoch, step, mode='only-reward')
                self.run_regular_training_step(self.state_model, tasks, epoch, step, mode='only-state')

    def run_meta_training_step(self, model, tasks, epoch, step, mode='joint'):

        assert mode in ['joint', 'only-reward']

        enc_data = [self.pre_adapt_replay_buffer.sample(self.num_sample_steps_for_adaptation, task_id=task_id) for
                    task_id in tasks]
        rb_data = [self.replay_buffer.sample(self.num_sample_steps_for_adaptation, task_id=task_id) for task_id in
                   tasks]

        feed_dict = self.get_feed_dict_for_training(model, enc_data, rb_data)

        _, meta_train_metrics, updated_contexts = self.sess.run([
            model.metatrain_op, model.meta_train_metrics, model.updated_contexts],
            feed_dict=feed_dict)

        if (step + 1) % self.num_training_steps_per_epoch == 0:
            # if step % self.num_training_steps_per_epoch == 0:
            self.log_avg_meta_training_metrics(epoch, meta_train_metrics, model, 'meta-model-')

        data_with_context = concatenate_traj_data([
            append_context_to_traj(traj_data, context) for traj_data, context in zip(rb_data, updated_contexts)
        ])
        self.sac_trainer._do_training(step, data_with_context)

    def run_regular_training_step(self, model, tasks, epoch, step, mode='only-state'):

        assert mode == 'only-state'

        train_data = [self.pre_adapt_replay_buffer.sample(self.num_sample_steps_for_adaptation, task_id=task_id) for
                      task_id in tasks]
        feed_dict = self.get_feed_dict_for_training(model, train_data)

        _, train_metrics = self.sess.run([model.train_op, model.train_metrics], feed_dict=feed_dict)

        if (step + 1) % self.num_training_steps_per_epoch == 0:
            self.logger.log_dict(epoch, avg_metrics_across_tasks(model, train_metrics['train_dicts'],
                                                                 model.metric_keys(train_dicts=True)),
                                 'regular-model-train')
            self.logger.log_dict(epoch,
                                 avg_metrics_across_tasks(model, train_metrics['val_dicts'], model.metric_keys()),
                                 'regular-model-val')

    def save_models_and_buffers(self, epoch):

        model_log_dir = self.log_dir + '/Itr_' + str(epoch) + '/'
        os.makedirs(model_log_dir, exist_ok=True)
        if self.joint_state_reward_model:
            self.model.save_model(model_log_dir + 'model.pkl')
        else:
            self.reward_model.save_model(model_log_dir + 'reward_model.pkl')
            self.state_model.save_model(model_log_dir + 'state_model.pkl')

        self.sac_trainer.save_model(model_log_dir)
        pickle.dump(self.pre_adapt_replay_buffer.convert_to_numpy(),
                    open(model_log_dir + 'pre_adapt_replay_buffer.pkl', 'wb'))
        pickle.dump(self.replay_buffer.convert_to_numpy(), open(model_log_dir + 'replay_buffer.pkl', 'wb'))

    def eval_during_training(self, epoch):

        def compute_returns(task_idxs):
            all_returns = []
            for task_id in task_idxs:
                self.sampler.reset_task(task_id)
                data = self.sampler.sample( self.num_sample_steps_for_adaptation, self.model.get_context())
                proc_data = prepare_data(data)
                updated_context = self.model.get_updated_context(proc_data[0], proc_data[1])
                all_returns.append(self.eval_single_task(epoch, updated_context, log_name = 'eval/perTask/Task_'+str(task_id)))
            return all_returns

        def log_returns(mode):
            task_idxs = np.arange(self.n_train_tasks) if mode == 'train' else \
                np.arange(self.n_train_tasks, self.n_train_tasks + self.n_val_tasks)

            avg_return = np.mean(compute_returns(task_idxs))
            print('epoch ' + str(epoch), mode+'_avg_return', avg_return)
            self.logger.log_dict(epoch, {'eval/'+mode+'_avg_return': avg_return})

        if self.eval_train_tasks:
            log_returns('train')
        if self.eval_val_tasks:
            log_returns('val')

    def log_avg_meta_training_metrics(self, epoch, all_metrics, model, log_prefix):

        self.logger.log_dict(epoch, all_metrics['meta_update_norms'], log_prefix + 'meta-update-norms/')

        all_train_dicts = []
        all_val_dicts = []
        for _step in range(model.fast_adapt_steps + 1):
            all_train_dicts.append(avg_metrics_across_tasks(model, all_metrics['train_dicts'],
                                                            model.metric_keys(train_dicts=True), _step))

            all_val_dicts.append(avg_metrics_across_tasks(model, all_metrics['val_dicts'],
                                                          model.metric_keys(), _step))

        self.logger.log_dict(epoch, dict_diff(all_train_dicts[0], all_train_dicts[-1]),
                             log_prefix + 'meta-train-improvement/')
        self.logger.log_dict(epoch, dict_diff(all_val_dicts[0], all_val_dicts[-1]),
                             log_prefix + 'meta-val-improvement/')

        if self.log_metrics_for_every_adapt_step:

            for _step in range(model.fast_adapt_steps + 1):
                self.logger.log_dict(epoch, all_train_dicts[_step], log_prefix + 'meta-train-step-' + str(_step) + '/')
                self.logger.log_dict(epoch, all_val_dicts[_step], log_prefix + 'meta-val-step-' + str(_step) + '/')

    def get_feed_dict_for_training(self, model, train_data, val_data=None):

        train_input, train_target = prepare_multitask_data(train_data)
        _phs = model.multi_task_placeholders
        if val_data == None:
            return {_phs[0]: train_input, _phs[1]: train_target}

        else:
            val_input, val_target = prepare_multitask_data(val_data)

            return {_phs[0]: train_input, _phs[1]: train_target,
                    _phs[2]: val_input, _phs[3]: val_target}

    ################################Extrapolation #################################################

    def extrapolate(self):
        '''
        Require : Train-phase data
        Algorithm:
        1. Take multiple fast update steps on the model initialization to obtain updated context  (using true data)
        2. Take states from train phase and current data , and use this to generate synthetic data. Add synthetic data
        to the model_buffer
        3. Use synthetic data to update policy
        4. Loop
        :return:
        '''

        context = self.model.get_context()
        self.collect_data(self.num_sample_steps_for_adaptation, self.task_id_for_extrapolation, context)

        if self.adapt_model_for_extrapolation:
            context = self.fast_adapt_model(self.model)
            print('adapted context', context)

        for epoch in range(self.num_extrapol_epochs):

            print('############## EPOCH ', epoch, ' ##################')
            self.eval_single_task(epoch, context)
            for step in range(self.num_sac_steps_per_epoch):
                if step % self.relabelling_interval == 0:
                    print('step ' + str(step))
                if self.relabel_data_for_extrapolation and step % self.relabelling_interval == 0:
                    if self.off_policy_relabelling:
                        self.off_policy_relabelling_single_task(context, self.task_id_for_extrapolation)
                    else:
                        self.rollout_model_single_task(context, self.task_id_for_extrapolation)

                for _ in range(self.num_sac_repeat_steps_for_extrapol):
                    data = self.get_sac_training_batch(self.batch_size, context)
                    self.sac_trainer._do_training(step, data)

    def fast_adapt_model(self, model):
        # run few adaptation steps for policy context, and conitnued adaptation for model fine-tuning
        inputs, targets = prepare_data(self.replay_buffer.sample(return_all_data=True))
        train_inputs, train_targets, val_inputs, val_targets = \
            split_data_into_train_val(inputs, targets, self.train_val_ratio)

        train_feed_dict = self._get_feed_dict_for_extrapolation((train_inputs, train_targets), model)
        val_feed_dict = self._get_feed_dict_for_extrapolation((val_inputs, val_targets), model)

        assert self.num_fast_adapt_steps_for_context > 0
        for fast_step in range(self.num_fast_adapt_steps_for_context):
            train_feed_dict[model.test_time_lr] = model.fast_adapt_lr if fast_step < model.fast_adapt_steps \
                else model.fast_adapt_lr / 10

            updated_context, model_loss_dict = self.sess.run([model.test_time_updated_context,
                                                              model.model_prior_loss_dict],
                                                             feed_dict=train_feed_dict)
            val_model_loss_dict = self.sess.run(model.model_prior_loss_dict, feed_dict=val_feed_dict)
            self.log_finetuning_model_losses(fast_step, model_loss_dict, val_model_loss_dict)

            self.sess.run(tf.assign(model.context, updated_context))
            context_for_policy = updated_context

        for fast_step in range(self.num_fast_adapt_steps_for_context,
                               self.num_fast_adapt_steps_for_context + self.num_extra_fast_adapt_steps_for_model):
            _, model_loss_dict = self.sess.run([model.test_task_train_op, model.model_prior_loss_dict],
                                               feed_dict=train_feed_dict)
            val_model_loss_dict = self.sess.run(model.model_prior_loss_dict, feed_dict=val_feed_dict)
            self.log_finetuning_model_losses(fast_step, model_loss_dict, val_model_loss_dict)

        return context_for_policy

    def off_policy_relabelling_single_task(self, context, task_id=None):

        obs, action, next_obs, term = self.sample_data_for_relabelling(task_id, only_states=False)
        if self.is_model_context_conditioned:
            _, rew, _, _ = self.fake_env.step(obs, action, context)
        else:
            _, rew, _, _ = self.fake_env.step(obs, action)
        self.model_buffer.add_traj(TrajData(obs, action, rew, next_obs, term))
 

    def rollout_model_single_task(self, context, task_id=None):

        obs = self.sample_data_for_relabelling(task_id, only_states = True)
        obs_with_context = append_context_to_array(obs, context)

        for i in range(self.model_rollout_length):
 
            action = self.sac_trainer._policy.actions_np(obs_with_context)
            next_obs, rew, term, info = self.fake_env.step(obs, action, context)
            
            if self.discard_term_states_while_relabelling:
                next_obs, rews, term, info = discard_term_states(obs, action, next_obs, rew, term)

            self.model_buffer.add_traj(TrajData(obs, action, rew, next_obs, term))

            nonterm_mask = ~term.squeeze(-1)
            if nonterm_mask.sum() == 0:
                print(
                    '[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                break
            obs = next_obs[nonterm_mask]

    def sample_data_for_relabelling(self, task_id=None, only_states=True):

        all_states = [];
        all_next_states = [];
        all_actions = [];
        all_terms = []
        tasks = np.random.choice(len(self.cross_task_data),
                                 self.num_cross_tasks_for_relabelling) if self.cross_task_relabelling_for_testing \
            else task_id * np.ones(self.num_cross_tasks_for_relabelling)
        print(tasks)
        for i, task in enumerate(tasks):
            idxs = np.random.choice(np.arange(self.cross_task_data_size),
                                    self.rollout_batch_size // self.num_cross_tasks_for_relabelling)
            all_states.append(self.cross_task_data[task]['observations'][idxs])
            if not only_states:
                all_actions.append(self.cross_task_data[task]['actions'][idxs])
                all_next_states.append(self.cross_task_data[task]['next_observations'][idxs])
                all_terms.append(self.cross_task_data[task]['terminals'][idxs])

        if only_states:
            return np.reshape(np.array(all_states), (-1, self.obs_dim))
        else:
            return np.reshape(np.array(all_states), (-1, self.obs_dim)), np.reshape(np.array(all_actions),
                                                                                    (-1, self.act_dim)), \
                   np.reshape(np.array(all_next_states), (-1, self.obs_dim)), np.reshape(np.array(all_terms), (-1, 1))

    def get_sac_training_batch(self, batch_size, context):

        if self.relabel_data_for_extrapolation:
            env_batch_size = int(batch_size * self.model_real_ratio)
            env_batch = sample_from_buffer_and_append_context(self.replay_buffer, env_batch_size, context)
            model_batch = sample_from_buffer_and_append_context(self.model_buffer, batch_size - env_batch_size, context)
            return concatenate_traj_data([env_batch, model_batch])

        else:
            return sample_from_buffer_and_append_context(self.replay_buffer, batch_size, context)

    def collect_data(self, num_env_samples, task_id, context):

        # self.sampler.reset_task(self.task_id_for_extrapolation)
        self.sampler.reset_task(task_id)
        data = self.sampler.sample(num_env_samples, context)

        if self.multi_task:
            self.replay_buffer.add(data, task_id=task_id)
        else:
            self.replay_buffer.add_traj(data)

    def eval_single_task(self, epoch, context, log_name = 'return'):

        eval_data = self.sampler.sample(self.max_path_length, context, max_episodes=1, deterministic=True)
        _ret = sum(eval_data.rewards)
        self.logger.log_dict(epoch, {log_name: _ret}, '')
        if log_name == 'return':
            print('avg_return', _ret)
        return _ret

    def _get_feed_dict_for_extrapolation(self, processed_data, model):

        train_input, train_target = processed_data
        _phs = model.single_task_placeholders

        return {_phs[0]: train_input, _phs[1]: train_target}

    def log_finetuning_model_losses(self, step, model_loss_dict, val_model_loss_dict):

        print_model_losses(step, model_loss_dict)
        self.logger.log_dict(step, model_loss_dict, 'fast_adapt_model_losses/')

        print_model_losses(step, val_model_loss_dict, 'val')
        self.logger.log_dict(step, val_model_loss_dict, 'fast_adapt_val_model_losses/')



