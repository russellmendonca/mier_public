from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from collections import OrderedDict
from copy import copy, deepcopy

import functools
import operator
import numpy as np
import pickle
import torch


def dict_diff(dict1, dict2):
    new_dict = {}
    for key in dict2:
        new_dict[key] = dict2[key] - dict1[key]
    return new_dict


def append_context_to_traj(traj_data, context):
    new_observations = append_context_to_array(traj_data.observations, context)
    new_next_observations = append_context_to_array(traj_data.next_observations, context)

    traj_data = deepcopy(traj_data)
    traj_data['observations'] = new_observations
    traj_data['next_observations'] = new_next_observations
    return traj_data

def append_context_to_array(_arr, context):
    num_points = np.shape(_arr)[0]
    context = np.ones((num_points, context.shape[-1])) * context
    return np.concatenate([_arr, context], axis=-1)

def prepare_data(all_env_data):
    obs = all_env_data.observations
    next_obs = all_env_data.next_observations
    actions = all_env_data.actions
    rewards = all_env_data.rewards.reshape(-1, 1)

    inputs = np.concatenate([obs, actions], axis=-1)
    targets = np.concatenate([rewards, next_obs - obs], axis=-1)
    idxs = np.random.permutation(np.arange(len(inputs)))

    return [inputs[idxs]], [targets[idxs]]

def sample_from_buffer_and_append_context(_buffer, batch_size, context):
    _batch = _buffer.sample(batch_size)
    _batch = append_context_to_traj(_batch, context)
    _batch.rewards = _batch.rewards.reshape(-1, 1)
    _batch.terminals = _batch.terminals.reshape(-1, 1)
    return _batch

def split_data_into_train_val(inputs, targets, train_val_ratio):
    inputs, targets = np.array(inputs), np.array(targets)
    num_points = np.shape(inputs)[1]
    num_train_points = int(num_points * train_val_ratio)
    random_idxs = np.random.permutation(np.arange(num_points))

    train_inputs = inputs[:, random_idxs[: num_train_points], :]
    train_targets = targets[:, random_idxs[: num_train_points], :]

    val_inputs = inputs[:, random_idxs[num_train_points:], :]
    val_targets = targets[:, random_idxs[num_train_points:], :]

    return train_inputs, train_targets, val_inputs, val_targets

def prepare_multitask_data(all_task_data):

    all_inputs = [] ; all_targets = []
    for task in range(len(all_task_data)):
        inputs , targets = prepare_data(all_task_data[task])
        all_inputs.append(inputs[0])
        all_targets.append(targets[0])

    return np.array(all_inputs), np.array(all_targets)

def to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    return np.array(value)


def to_torch(value, device='cpu'):
    if isinstance(value, torch.Tensor):
        value = value.to(device)
        if value.requires_grad:
            value = value.detach()
        return value
    return torch.tensor(value, device=device, dtype=torch.float32)


class TrajData(object):

    def __init__(self, observations, actions, rewards, next_observations, terminals):
        self.observations = np.array(observations, np.float32)
        self.actions = np.array(actions, np.float32)
        self.rewards = np.array(rewards, np.float32)
        self.next_observations = np.array(next_observations, np.float32)
        self.terminals = np.array(terminals, np.float32)

    def torch(self, device='cpu'):
        return TorchTrajData(self, device)

    def __getitem__(self, key):
        assert key in self.data_keys
        return getattr(self, key)

    def __setitem__(self, name, value):
        assert name in self.data_keys
        setattr(self, name, to_numpy(value))

    def __len__(self):
        return int(self.observations.shape[0])

    def append_context(self, context):
        context = to_numpy(context).astype(np.float32).reshape(1, -1)
        context = np.tile(context, (self.observations.shape[0], 1))
        new_data = copy(self)
        new_data.observations = np.concatenate([new_data.observations, context], axis=1)
        new_data.next_observations = np.concatenate([new_data.next_observations, context], axis=1)
        return new_data

    @property
    def data_keys(self):
        return (
            'observations', 'actions', 'rewards', 'next_observations', 'terminals'
        )

    def sample(self, size):
        indices = np.random.choice(len(self), size, replace=size > len(self))
        data_dict = {}
        for key in self.data_keys:
            if self[key] is not None:
                data_dict[key] = self[key][indices, ...]
            else:
                data_dict[key] = None

        return TrajData(**data_dict)


class TorchTrajData(object):

    def __init__(self, traj_data, device='cpu'):
        self._device = device
        self._data_keys = traj_data.data_keys
        for key in traj_data.data_keys:
            if getattr(traj_data, key) is not None:
                data = to_torch(getattr(traj_data, key), device)
            else:
                data = None

            setattr(self, key, data)

    def __getitem__(self, key):
        assert key in self.data_keys
        return getattr(self, key)

    def __setitem__(self, name, value):
        assert name in self.data_keys

        setattr(
            self, name,
            to_torch(value, self.device)
        )

    def append_context(self, context):
        context = to_numpy(context).astype(np.float32).reshape(1, -1)
        context = to_torch(context, self._device).repeat(self.observations.shape[0], 1)
        new_data = copy(self)
        new_data.observations = torch.cat([new_data.observations, context], dim=1)
        new_data.next_observations = torch.cat([new_data.next_observations, context], dim=1)
        return new_data

    @property
    def data_keys(self):
        return self._data_keys

    @property
    def device(self):
        return self._device

    def __len__(self):
        return int(self.observations.shape[0])


def concat_observations_context(observations, context):
    if isinstance(observations, torch.Tensor):
        context = context.detach()
        if len(observations.shape) == 1:
            # Single observation and goal
            return torch.cat([observations, context], dim=0)
        else:
            context = context.repeat(observations.shape[0], 1)
            return torch.cat([observations, context], dim=1)
    else:
        if len(observations.shape) == 1:
            return np.concatenate([observations, context], axis=0)
        else:
            context = np.tile(context, [observations.shape[0], 1])
            return np.concatenate([observations, context], axis=1)


def concatenate_traj_data(trajs):
    keys = trajs[0].data_keys
    is_torch = isinstance(trajs[0], TorchTrajData)

    new_data = copy(trajs[0])

    for key in keys:
        if is_torch:
            new_data[key] = torch.cat(
                [traj[key] for traj in trajs], dim=0
            )
        else:
            new_data[key] = np.concatenate(
                [traj[key] for traj in trajs], axis=0
            )
    return new_data


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._max_size = size
        self._next_idx = 0
        self._size = 0
        self._initialized = False

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._terminals = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add(self, obs_t, action, reward, obs_tp1, done):
        if not self._initialized:
            self._init_storage(obs_t.size, action.size)

        self._observations[self._next_idx, :] = np.array(obs_t, dtype=np.float32).flatten()
        self._next_observations[self._next_idx, :] = np.array(obs_tp1, dtype=np.float32).flatten()
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32).flatten()
        self._rewards[self._next_idx] = reward
        self._terminals[self._next_idx] = float(done)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size

    def add_traj(self, traj_data):

        for o, a, r, no, d in zip(traj_data.observations, traj_data.actions,
                                  traj_data.rewards, traj_data.next_observations,
                                  traj_data.terminals):
            self.add(o, a, r, no, d)

    def sample(self, batch_size=256, return_all_data=False):
        try:
            if return_all_data:
                indices = np.arange(0, len(self))
            else:
                indices = np.random.randint(0, len(self), batch_size)
        except:
            raise AssertionError('Buffer Error')

        return TrajData(
            self._observations[indices, ...],
            self._actions[indices, ...],
            self._rewards[indices, ...],
            self._next_observations[indices, ...],
            self._terminals[indices, ...]
        )


class MultiTaskReplayBuffer(object):
    """Muti-task replay buffer with samples seperated by tasks."""

    def __init__(self, max_buffer_size, num_tasks=100):

        self.max_buffer_size = max_buffer_size
        self.buffers = OrderedDict()
        for i in range(num_tasks):
            self.buffers[i] = ReplayBuffer(self.max_buffer_size)

    def convert_to_numpy(self):

        keys = ["observations", "actions", "rewards", "next_observations", "terminals"]
        def helper(key, task_id=None):

            return self.sample(task_id=task_id, return_all_data=True)[key]

        def get_task_data(task_id=None):
            task_data = {}
            for key_name, _val in zip(keys, [helper(key, task_id) for key in keys]):
                task_data[key_name] = _val
            return task_data

        all_task_data = {}
        for id in range(len(self.buffers)):
            all_task_data[id] = get_task_data(id)

        return all_task_data

    def load_data(self, load_path):
        keys = ["observations", "actions", "rewards", "next_observations", "terminals"]
        data = pickle.load(open(load_path, 'rb'))
        for task_id in range(len(data)):
            task_data = data[task_id]
            obs, acts, rews, next_obs, terms = [task_data[key] for key in keys]
            self.add(TrajData(obs, acts, rews, next_obs, terms), task_id)

    def add(self, traj_data, task_id=None):

        assert type(task_id) != None
        task_id = int(task_id)
        replay_buffer = self.buffers[task_id]
        replay_buffer.add_traj(traj_data)
        return task_id

    def sample(self, batch_size=256, task_id=None, device='cpu', torch=False, return_all_data=False):

        assert type(task_id) != None
        data = self.buffers[task_id].sample(batch_size, return_all_data)
        if torch:
            data = data.torch(device)
        return data

    def remove_zeros(self):
        new_multiTask_buffer = deepcopy(self)

        for idx in new_multiTask_buffer.buffers:

            buffer = new_multiTask_buffer.buffers[idx]
            if buffer._size < self.max_buffer_size:
                last_idx = buffer._next_idx
                buffer._observations = buffer._observations[:last_idx]
                buffer._next_observations = buffer._next_observations[:last_idx]
                buffer._actions = buffer._actions[:last_idx]
                buffer._rewards = buffer._rewards[:last_idx]
                buffer._terminals = buffer._terminals[:last_idx]
                new_multiTask_buffer.buffers[idx] = buffer

        return new_multiTask_buffer

    @property
    def size(self):
        return sum([len(r) for r in self.buffers.values()])

    @property
    def total_time_steps_added(self):
        return self._total_time_steps_added

    @property
    def tasks(self):
        return list(self.buffers.keys())

    def __getitem__(self, key):
        return self.buffers[key]

    def __iter__(self):
        return iter(self.buffers)

    def __len__(self):
        return len(self.buffers)


class DoubleMultiTaskReplayBuffer(object):
    def __init__(self, max_buffer_size, num_tasks=100):
        self._pre_adapt_replay_buffer = MultiTaskReplayBuffer(
            max_buffer_size, num_tasks
        )
        self._post_adapt_replay_buffer = MultiTaskReplayBuffer(
            max_buffer_size, num_tasks
        )

    def add_pre_adapt(self, *args, **kwargs):
        return self.pre_adapt_replay_buffer.add(*args, **kwargs)

    def add_post_adapt(self, *args, **kwargs):
        return self.post_adapt_replay_buffer.add(*args, **kwargs)

    def sample_pre_adapt(self, *args, **kwargs):
        return self.pre_adapt_replay_buffer.sample(*args, **kwargs)

    def sample_post_adapt(self, *args, **kwargs):
        return self.post_adapt_replay_buffer.sample(*args, **kwargs)

    @property
    def pre_adapt_replay_buffer(self):
        return self._pre_adapt_replay_buffer

    @property
    def post_adapt_replay_buffer(self):
        return self._post_adapt_replay_buffer

    @property
    def tasks(self):
        return self.pre_adapt_replay_buffer.tasks

    @property
    def size(self):
        return (self.pre_adapt_replay_buffer.size
                + self.post_adapt_replay_buffer.size)

    @property
    def total_time_steps_added(self):
        return (self.pre_adapt_replay_buffer.total_time_steps_added
                + self.post_adapt_replay_buffer.total_time_steps_added)

    def __len__(self):
        return max(
            len(self.pre_adapt_replay_buffer),
            len(self.post_adapt_replay_buffer)
        )
