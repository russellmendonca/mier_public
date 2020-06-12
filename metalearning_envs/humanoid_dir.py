import numpy as np
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv

from . import register_env


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


@register_env('humanoid-dir')
class HumanoidDirEnv(HumanoidEnv):

    def __init__(self,  n_tasks, restricted_train_set=False, lin_vel_factor=0.25):
        self.restricted_train_set = restricted_train_set
        self.lin_vel_factor = lin_vel_factor
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)
        super(HumanoidDirEnv, self).__init__()

    def step(self, action):
        pos_before = np.copy(mass_center(self.model, self.sim)[:2])
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[:2]

        alive_bonus = 5.0
        data = self.sim.data
        goal_direction = (np.cos(self._goal), np.sin(self._goal))
        lin_vel_cost = self.lin_vel_factor * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        xy_pos = self.sim.data.qpos.flat[:2]
        return self._get_obs(), reward, done, dict(xy_pos = xy_pos,
                                                   reward_linvel=lin_vel_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']  # assume parameterization of task by single vector

    def sample_tasks(self, num_tasks):
        # velocities = np.random.uniform(0., 1.0 * np.pi, size=(num_tasks,))
        if self.restricted_train_set:
            assert num_tasks == 110
            train_directions = np.random.uniform(np.pi/6, np.pi/3, size=(100,))
            val_directions = np.concatenate([np.linspace(0, np.pi/6, 5), np.linspace(np.pi/3, np.pi/2, 5)], axis =0)
            directions = np.concatenate([train_directions, val_directions], axis=0)
        else:
            directions = np.random.uniform(0, 2*np.pi, size=(num_tasks,))
          
        tasks = [{'goal': d} for d in directions]
        return tasks

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        z = next_obs[:, 0]
        done = (z < 1.0) + (z > 2.0)
        #
        done = done[:, None]
        return done
