import numpy as np
import tensorflow as tf
from data import append_context_to_array


class FakeEnv:

    def __init__(self, model_dict, config, joint_state_reward_model=True):
        self.joint_state_reward_model = joint_state_reward_model
        if self.joint_state_reward_model:
            self.model = model_dict['model']
        else:
            self.state_model = model_dict['state_model']
            self.reward_model = model_dict['reward_model']

        self.config = config

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (
                k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def get_prediction(self, inputs, factored):

        if self.joint_state_reward_model:
            return self.model.predict(inputs, factored)
        else:
            means1, vars1 = self.reward_model.predict(inputs, factored )
            means2, vars2 = self.state_model.predict(inputs, factored)

            return np.concatenate([means1[:, :1], means2[:, 1:]], axis=1), \
                   np.concatenate([vars1[:, :1], vars2[:, 1:]], axis=1)

    def step(self, obs, act):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)

        model_means, model_vars = self.get_prediction(inputs, factored = True)

        # if self.model.pred_dynamics:
        model_means[:, 1:] += obs
        model_stds = np.sqrt(model_vars)
        deterministic = self.model.is_deterministic if self.joint_state_reward_model \
                        else self.state_model.is_deterministic
        if deterministic:
            samples = model_means
        else:
            samples = model_means + np.random.normal(
                size=model_means.shape) * model_stds

        log_prob, dev = self._get_logprob(samples, model_means, model_vars)

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self.config.termination_fn(obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info

    def close(self):
        pass
