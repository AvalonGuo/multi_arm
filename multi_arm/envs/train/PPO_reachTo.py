import numpy as np
from typing import Optional
import pybullet as p

import math
from multi_arm.pybullet import PyBullet
from multi_arm.envs.core import RobotTaskEnv
from multi_arm.envs.robots.panda import Panda
from multi_arm.envs.tasks.reachTo import ReachTo

import torch as th 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env



class ReachToPPOEnv(RobotTaskEnv):
    def __init__(
        self,
        reward_type: str = "dense",
        control_type: str = "ee",
    ) -> None:
        sim = PyBullet(render=True)
        robot = Panda(sim,block_gripper=True, base_position=np.array([-0.5, 0.0, 0.0]), control_type=control_type)
        task = ReachTo(sim,get_ee_position=robot.get_ee_position ,reward_type=reward_type)
        super().__init__(
            robot,
            task
        )

env = ReachToPPOEnv()


# total training timesteps
total_timesteps = 26000*5
# Learning rate for the PPO optimizer
learning_rate = 0.001
# Batch size for training the PPO model
batch_size = 64
# Number of hidden units for the policy networkzl
pi_hidden_units = [64, 64]
# Number of hidden units for the value function network
vf_hidden_units = [64, 64]


tb_log_name = f"ReachTo_PPO_{learning_rate}" 
model_save_name = f"ReachTo_PPO_{learning_rate}.zip"


policy_kwargs = dict(
    activation_fn=th.nn.ReLU, net_arch=[dict(pi=pi_hidden_units, vf=vf_hidden_units)]
)

#############################
# Set up PPO model
# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.MultiInputPolicy
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="runs",
    batch_size=batch_size,
    normalize_advantage=True,
    learning_rate=learning_rate,
    policy_kwargs=policy_kwargs,
)

#############################
# Train agent
model.learn(
    total_timesteps=total_timesteps,
    tb_log_name=tb_log_name,
)

#############################
# Save trained model
model.save(model_save_name)



