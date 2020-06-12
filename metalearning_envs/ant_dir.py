import numpy as np
from .ant_multitask_base import MultitaskAntEnv
from . import register_env

@register_env('ant-dir')
class AntDirEnv(MultitaskAntEnv):

    def __init__(self, n_tasks=2, task_mode = 'forward_backward'):
     
        self.task_mode = task_mode
        super(AntDirEnv, self).__init__({}, n_tasks)

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2] / self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def sample_tasks(self, n_tasks):

        if self.task_mode == 'forward_backward':
            assert n_tasks == 2
            directions = np.array([0., np.pi])
             
        elif self.task_mode == 'ood':

            assert n_tasks == 110
            np.random.seed(42)
            train_tasks =  np.random.uniform(0, 1.5*np.pi, (100))
            test_tasks =   np.linspace(1.5*np.pi, 2 * np.pi, (10))
            directions =  np.concatenate([train_tasks, test_tasks])
        
        elif self.task_mode == 'ood_sensitivity':
            directions = np.linspace(1.5*np.pi, 2*np.pi, 10)

        tasks = [{'goal': direction} for direction in directions]
        return tasks
 

if __name__ == '__main__':
    env =  AntDirEnv()
    for idx in range(2):
        env.reset()
        env.reset_task(idx)
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()
