import numpy as np
from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv

from . import register_env

@register_env('walker-rand-params')
class WalkerRandParamsWrappedEnv(Walker2DRandParamsEnv):
    def __init__(self, n_tasks=60, task_mode='standard'):

        self.task_mode = task_mode
        super(WalkerRandParamsWrappedEnv, self).__init__(log_scale_limit = 3.0)
        
        if self.task_mode == 'standard':
            self.tasks = self.sample_tasks(n_tasks)
        elif self.task_mode == 'ood':
            self.tasks = self.get_train_and_ood_tasks(n_tasks)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()

    def get_train_and_ood_tasks(self, n_tasks):
        assert n_tasks == 60
        np.random.seed(42)    
        tasks = self.sample_tasks(40)
        task_keys = tasks[0].keys() 
        ood_param_dist = {}
        for key in task_keys:
            ood_param_dist[key+'_std'] =   np.std(np.array([task[key] for task in tasks]), axis = 0)
            ood_param_dist[key+'_mean'] =  np.mean(np.array([task[key] for task in tasks]), axis = 0) + 5*ood_param_dist[key+'_std']
        
        for new_task_idx in range(20):
            new_task = {}
            for key in task_keys:
                new_task[key] = np.abs(np.random.normal(ood_param_dist[key+'_mean'], ood_param_dist[key+'_std']))
            tasks.append(new_task)

        return tasks

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = (height > 0.8) \
                   * (height < 2.0) \
                   * (angle > -1.0) \
                   * (angle < 1.0)
        done = ~not_done
        done = done[:, None]
        return done


if __name__ == '__main__':
    env = WalkerRandParamsWrappedEnv(task_mode = 'ood')

    import ipdb ; ipdb.set_trace()
    def get_rep(env):
        tasks = env.tasks
        task_keys = tasks[0].keys()
        representative_values = {}
        for key in tasks[0].keys():
            representative_values[key+'-mean'] =  np.mean(np.array([task[key] for task in tasks]), axis = 0)
            representative_values[key+'-std'] =   np.std(np.array([task[key] for task in tasks]), axis = 0)
        return representative_values
    
    rep_limited = get_rep(WalkerRandParamsWrappedEnv(restricted_train_set=True))
    rep_std     = get_rep(WalkerRandParamsWrappedEnv(restricted_train_set=False))
    import ipdb ; ipdb.set_trace()
    asdad = 3


