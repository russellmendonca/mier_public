import os
from os import path

import mujoco_py
import numpy as np
from gym.envs.mujoco import mujoco_env
from .serializable import Serializable

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')
class MujocoEnv(mujoco_env.MujocoEnv, Serializable):
    """
    My own wrapper around MujocoEnv.

    The caller needs to declare
    """

    def __init__(
            self,
            model_path,
            frame_skip=1,
            model_path_is_local=True,
            automatically_set_obs_and_action_space=False,
    ):
        if model_path_is_local:
            model_path = get_asset_xml(model_path)
        if automatically_set_obs_and_action_space:
            mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip)
        else:
            """
            Code below is copy/pasted from MujocoEnv's __init__ function.
            """
            if model_path.startswith("/"):
                fullpath = model_path
            else:
                fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
            if not path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
            self.frame_skip = frame_skip
            self.model = mujoco_py.load_model_from_path(fullpath)
            self.sim = mujoco_py.MjSim(self.model)
            self.data = self.sim.data
            self.viewer = None

            self.metadata = {
                'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': int(np.round(1.0 / self.dt))
            }

            self.init_qpos = self.sim.data.qpos.ravel().copy()
            self.init_qvel = self.sim.data.qvel.ravel().copy()
            self.seed()

    def init_serialization(self, locals):
        Serializable.quick_init(self, locals)

    def log_diagnostics(self, paths):
        pass

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.data.body_xmat[idx].reshape((3, 3))

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

def get_asset_xml(xml_name):
    return os.path.join(ENV_ASSET_DIR, xml_name)
