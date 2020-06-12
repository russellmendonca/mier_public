from rand_param_envs.hopper_rand_params import HopperRandParamsEnv

from . import register_env


@register_env('hopper-rand-params')
class HopperRandParamsWrappedEnv(HopperRandParamsEnv):
    def __init__(self, n_tasks=2, restricted_train_set=False):
        log_scale_limit = 1.0 if restricted_train_set else 3.0
        super(HopperRandParamsWrappedEnv, self).__init__(log_scale_limit = log_scale_limit)
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = np.isfinite(next_obs).all(axis=-1) \
                   * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                   * (height > .7) \
                   * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:, None]
        return done
