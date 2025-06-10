import random
import math

import numpy as np
import torch
import torch.nn as nn
from torchrl.envs import EnvBase
from torchrl.data import Unbounded, Composite, Bounded, Binary
from tensordict import TensorDict
import genesis as gs


class Environment(EnvBase):
    def __init__(self):
        super().__init__()
        #Agentの観測空間
        self.observation_spec = Composite(
            normalized_ball_angle = Bounded(low=-1, high=1, shape=(1,), dtype=torch.float32),
            normalized_enemy_goal_angle = Bounded(low=-1, high=1, shape=(1,), dtype=torch.float32),
            is_online = Binary(shape=(1,), dtype=torch.float32),
        )
        #Agentの行動空間
        self.action_space = Composite(
            normalized_x_axis = Bounded(low=-1, high=1, shape=(1,), dtype=torch.float32),
            normalized_y_axis = Bounded(low=-1, high=1, shape=(1,), dtype=torch.float32),
        )

    def _reset(self, tensordict):
        pass


    def _step(self, tensordict):
        out = TensorDict(
            {
                "reward": torch.tensor([0.0], dtype=torch.float32),
                "done": torch.tensor([False], dtype=torch.bool),
            },
            batch_size=tensordict.batch_size,
        )
        return out


    def _set_seed(self):
        pass
