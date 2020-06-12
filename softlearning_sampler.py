import numpy as np

import gym

import multiprocessing as mp

from data import TrajData, concatenate_traj_data

from misc_utils import unwrapped_env


class EnvMaker(object):

	def __init__(self, env_class, *args, **kwargs):
		self.env_class = env_class
		self.args = args
		self.kwargs = kwargs

	def __call__(self):
		return self.env_class(*args, **kwargs)


class GymEnvMaker(object):

	def __init__(self, env_name):
		self.env_name = env_name

	def __call__(self):
		return gym.make(self.env_name)

class ContextConditionedSimpleSampler(object):
	
	def __init__(self, env, policy , max_path_length ,  exploration_policy = None):
		# self.env_fn = env_fn
		self._env = env
		self.policy = policy
		self.exploration_policy = exploration_policy if (exploration_policy is not None) else policy
		self.max_path_length = max_path_length

		self._path_length = 0
		self._current_observation = None

	def sample(self, max_samples, context=None, deterministic=False, max_episodes = None , exploration = False,
	           return_infos = False):

		policy = self.exploration_policy if exploration else self.policy

		assert max_samples < np.inf, "max_samples  must be finite"
		if context is not None:
			context = context.flatten()

		trajData_lst = []; infos_across_trajs = []
		n_steps_total = 0 ; num_episodes = 0

		with policy.set_deterministic(deterministic):
			while n_steps_total < max_samples:

				observations, actions, rewards, next_observations, dones, all_infos = \
					self.rollout(context, policy,  remaining_samples=max_samples - n_steps_total)

				trajData_lst.append(TrajData(observations, actions, rewards, next_observations, dones))
				infos_across_trajs.append(all_infos)
				n_steps_total += len(observations)
				num_episodes+=1
				if max_episodes and num_episodes>=max_episodes:
					break
		if return_infos:
			return concatenate_traj_data(trajData_lst), infos_across_trajs
		else:
			return concatenate_traj_data(trajData_lst)

	def rollout(self, context, policy, remaining_samples):

		observations = []
		actions = []
		rewards = []
		next_observations = []
		dones = []
		all_infos = []
		remaining_samples = remaining_samples
		obs = self._env.reset()

		for _ in range(self.max_path_length):
			if context is  None:
				action = policy.actions_np([obs[None]])[0]
			else:
				action = policy.actions_np([ np.concatenate([obs, context])[None] ])[0]


			next_obs, reward, done, infos = self._env.step(action)
			#img = self._env.render()
			observations.append(obs)
			actions.append(action)
			rewards.append(reward)
			dones.append(done)
			next_observations.append(next_obs)
			all_infos.append(infos)
			remaining_samples -= 1
			if done or remaining_samples<=0:
			#if done:
				policy.reset()
				break
			obs = next_obs

		return observations, actions, rewards, next_observations, dones, all_infos


	def reset_task(self, task):
		return self.unwrapped_env.reset_task(task)

	def sample_tasks(self, num_tasks):
		return self.unwrapped_env.sample_tasks(num_tasks)

	@property
	def env(self):
		return self._env

	@property
	def unwrapped_env(self):
		return unwrapped_env(self._env)
