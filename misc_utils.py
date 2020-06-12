import random
import io
import os
import csv
import pprint
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from gym import wrappers


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def discard_term_states(obs, action, next_obs, rew, term):
    idxs = [i for i in range(term.shape[0]) if term[i][0] == False]
    return obs[idxs], action[idxs], next_obs[idxs], rew[idxs], term[idxs]

def get_sep_model_hyperparams(model_hyperparams, meta_learn_state_dynamics, meta_learn_reward):
    state_model_hyperparams = model_hyperparams.copy()
    reward_model_hyperparams = model_hyperparams.copy()

    for key, val in zip(['name', 'rew_pred', 'meta_train'],
                        ['stateModel', False, meta_learn_state_dynamics]):
        state_model_hyperparams[key] = val

    for key, val in zip(['name', 'state_pred', 'meta_train'],
                        ['rewardModel', False, meta_learn_reward]):
        reward_model_hyperparams[key] = val

    return state_model_hyperparams, reward_model_hyperparams

def print_model_losses(step, model_loss_dict, prefix=''):

    print('step', str(step), prefix+' total_state_loss', str(model_loss_dict['total_state_loss']),
          prefix+' total_rew_loss', str(model_loss_dict['total_rew_loss']))

def avg_metrics_across_tasks(model, multi_task_dict, key_list, _step=None):
    proc_dict = {}
    for key in key_list:
        # for key in ['mse_rew_loss']:
        if _step == (model.fast_adapt_steps) and key == 'grad_norm':
            pass
        elif _step is not None:
            proc_dict[key] = np.mean(
                [multi_task_dict[i][_step][key] for i in range(model.meta_batch_size)])
        else:
            proc_dict[key] = np.mean([multi_task_dict[i][key] for i in range(model.meta_batch_size)])

    return proc_dict

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


def load_data(data_path, multi_task = False, key = None):
    if key:
        loaded_data = pickle.load(open(data_path, 'rb'))[key]
    else:
        loaded_data = pickle.load(open(data_path, 'rb'))

    if multi_task:
        loaded_data_size = min([len(loaded_data[task]['observations']) \
              for task in range(len(loaded_data))])
    else:
        loaded_data_size = len(loaded_data['observations'])

    return loaded_data, loaded_data_size

def set_random_seed(seed):
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    random.seed(seed)


def direct_logging(data, output_dir):
    # import ipdb ; ipdb.set_trace()
    for metric in data:
        metric_dir = output_dir + metric
        if os.path.isdir(metric_dir) != True:
            os.makedirs(metric_dir, exist_ok=True)

        with open(metric_dir + '/progress.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([data[metric]])
        csvFile.close()


class TensorBoardLogger(object):
    """Logging to TensorBoard outside of TensorFlow ops."""

    def __init__(self, output_dir):
        if not tf.gfile.Exists(output_dir):
            tf.gfile.MakeDirs(output_dir)
        self.output_dir = output_dir
        self.file_writer = tf.summary.FileWriter(output_dir)

    def log_scaler(self, step, name, value):
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=name, simple_value=value)]
        )
        self.file_writer.add_summary(summary, step)

    def log_image(self, step, name, image):
        summary = tf.Summary(
            value=[tf.Summary.Value(
                tag=name,
                image=self._make_image(image)
            )]
        )
        self.file_writer.add_summary(summary, step)

    def log_images(self, step, data):
        if len(data) == 0:
            return
        summary = tf.Summary(
            value=[
                tf.Summary.Value(tag=name, image=self._make_image(image))
                for name, image in data.items() if image is not None
            ]
        )
        self.file_writer.add_summary(summary, step)

    def _make_image(self, tensor):
        """Convert an numpy representation image to Image protobuf"""
        height, width, channel = tensor.shape
        image = Image.fromarray(tensor)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(
            height=height,
            width=width,
            colorspace=channel,
            encoded_image_string=image_string
        )

    def add_name_prefix_to_dict(self, _dict, prefix):
        new_dict = {}
        for key in _dict:
            new_dict[prefix + key] = _dict[key]
        return new_dict

    def log_dict(self, step, data, name_prefix=''):

        data = self.add_name_prefix_to_dict(data, name_prefix)
        summary = tf.Summary(
            value=[
                tf.Summary.Value(tag=name, simple_value=value)
                for name, value in data.items() if value is not None
            ]
        )

        direct_logging(data, os.path.join(self.output_dir, 'logs/'))
        self.file_writer.add_summary(summary, step)

    def flush(self):
        self.file_writer.flush()


def unwrapped_env(env):
    if isinstance(env, wrappers.TimeLimit) \
            or isinstance(env, wrappers.Monitor) \
            or isinstance(env, wrappers.FlattenDictWrapper):
        return env.unwrapped
    return env


def average_metrics(metrics):
    if len(metrics) == 0:
        return {}
    new_metrics = {}
    for key in metrics[0].keys():
        new_metrics[key] = np.mean([m[key] for m in metrics])

    return new_metrics


def print_flags(flags, flags_def):
    logging.info(
        'Running training with hyperparameters: \n{}'.format(
            pprint.pformat(
                ['{}: {}'.format(key, getattr(flags, key)) for key in flags_def]
            )
        )
    )

def parse_network_arch(arch):
    if len(arch) == 0:
        return []
    return [int(x) for x in arch.split('-')]

def consolidate_multiple_task_buffers(exp_dir, suffix, num_tasks =10, epoch=240):
    import pickle
    all_task_data = {}
    for task in range(num_tasks):
        all_task_data[task] = pickle.load(open(exp_dir+'task_'+str(task)+suffix +'/epoch_'+str(epoch)+'.pkl', 'rb'))
    os.makedirs(exp_dir+'train_tasks_replay_buffer', exist_ok = True)
    pickle.dump({'replay_buffer': all_task_data}, open(exp_dir+'train_tasks_replay_buffer/epoch_'+str(epoch)+'.pkl', 'wb'))

def make_task_randomized_buffer(task_sep_buffer, save_dir, num_tasks = 10):
    import pickle
    keys = ["observations", "actions", "rewards", "next_observations", "terminals"]
    data = pickle.load(open(task_sep_buffer, 'rb'))['replay_buffer']
    size = sum([len(data[task]["observations"]) for task in range(num_tasks)])
    idxs = np.random.permutation(size)
    def helper(key):
        return np.concatenate([data[task][key] for task in range(num_tasks) ])[idxs]

    randomized_data ={}
    for key_name, _val in zip(keys, [helper(key) for key in keys]):
        randomized_data[key_name] = _val
    pickle.dump(randomized_data, open(save_dir + '/task_randomized_data.pkl', 'wb'))


def consolidate_pearl_enc_buffers(_buff_path, start_epoch, end_epoch, gap, num_tasks = 10):
    from data import MultiTaskReplayBuffer , TrajData
    consolidated_buffer_data =MultiTaskReplayBuffer(int(1e6), num_tasks)

    def add_buffer_data(buffer_data):

        for task_id in range(num_tasks):
            td = buffer_data[task_id]
            traj_data = TrajData(td['observations'], td['actions'], td['rewards'], td['next_observations'], td['terminals'])
            consolidated_buffer_data.add(traj_data, task_id)

    for epoch in range(start_epoch, end_epoch + 1, gap):
        buffer_data = pickle.load(open(_buff_path+'epoch_'+str(epoch)+'.pkl', 'rb'))['enc_replay_buffer']
        add_buffer_data(buffer_data)

    all_data = {}
    all_data['replay_buffer'] = consolidated_buffer_data.convert_to_numpy()
    pickle.dump(all_data, open(_buff_path+'consolidated_enc_buffer.pkl', 'wb'))
