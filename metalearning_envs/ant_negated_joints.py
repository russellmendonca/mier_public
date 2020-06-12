import numpy as np
from itertools import combinations
from .ant import AntEnv
from . import register_env
@register_env('ant-negated-joints')
class AntModControl(AntEnv):

    def __init__(self, n_tasks):
        assert n_tasks == 50
        # 35 train tasks, 35 test tasks
        self.tasks = gen_neg_tasks()
        self.mask = self.tasks[0].get('mask')
        super(AntModControl, self).__init__()

    def step(self, action):

        action = self.mask*action
        torso_xyz_before = np.array(self.get_body_com("torso"))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = torso_velocity[0] / self.dt

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

    def reset_task(self, idx):

        self._task = self.tasks[idx]
        self.mask = self._task['mask']
        return self.reset()


def gen_neg_tasks():
    # 35 train tasks, followed by 15 test tasks
    all_tasks = []
    all_train_neg_idxs = list(combinations(np.arange(7), 4))

    for i, neg_idxs in enumerate(all_train_neg_idxs):
        mask = np.ones(8)
        for idx in neg_idxs:
            mask[idx] = -1
        all_tasks.append({'mask': mask})

    all_test_neg_idxs = list(combinations(np.arange(7), 3))[:15]
    for i, neg_idxs in enumerate(all_test_neg_idxs):
        mask = np.ones(8)
        mask[-1] = -1
        for idx in neg_idxs:
            mask[idx] = -1
        all_tasks.append({'mask': mask})

    return all_tasks
