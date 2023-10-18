import numpy as np
from typing import Optional
import pybullet as p
import math
from multi_arm.pybullet import PyBullet
from multi_arm.envs.core import RobotTaskEnv
from multi_arm.envs.my_core import MyRobotTaskEnv
from multi_arm.envs.robots.panda import Panda
from multi_arm.envs.tasks.Multi_Reach import Multi_Reach

import torch as th 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class InserationPPOEnv(MyRobotTaskEnv):
    def __init__(
        self,
        reward_type: str = "dense",
        control_type: str = "ee",
    ) -> None:
        self.sim = PyBullet(render=False
        
        )
        robot1 = Panda(self.sim,body_name="panda",block_gripper=True, base_position=np.array([-0.6, 0.5, 0.0]), control_type=control_type)
        robot2 = Panda(self.sim,body_name="panda2",block_gripper=True,base_position=np.array([-0.5, -0.5, 0.0]),control_type=control_type)
        robot3 = Panda(self.sim,body_name="panda3",block_gripper=True, base_position=np.array([0.5, -0.5, 0.0]), control_type=control_type)
        robot4 = Panda(self.sim,body_name="panda4",block_gripper=True,base_position=np.array([0.4, 0.5, 0.0]),control_type=control_type)
        self.sim.set_base_pose(body="panda",position= np.array([-0.6, 0.5, 0.0]),orientation= np.array([0,0,-math.pi/2]))
        self.sim.set_base_pose(body="panda2",position= np.array([-0.5,-0.5, 0.0]),orientation= np.array([0,0,math.pi/2]))
        self.sim.set_base_pose(body="panda3",position= np.array([0.5, -0.5, 0.0]),orientation= np.array([0,0,math.pi/2]))
        self.sim.set_base_pose(body="panda4",position= np.array([0.4,0.5, 0.0]),orientation= np.array([0,0,-math.pi/2]))
        robots = [robot1,robot2,robot3,robot4]
        task = Multi_Reach(sim=self.sim,robots=robots,reward_type=reward_type)
        super().__init__(
            robots,
            task
        )

env = InserationPPOEnv()
# total training timesteps
total_timesteps = 20000*50
# Learning rate for the PPO optimizer
learning_rate = 0.001
# Batch size for training the PPO model
batch_size = 64
# Number of hidden units for the policy networkzl
pi_hidden_units = [64, 64]
# Number of hidden units for the value function network
vf_hidden_units = [64, 64]
# Custom actor (pi) and value function (vf) networks
# of two layers of size 64 each with Relu activation function
policy_kwargs = dict(
    activation_fn=th.nn.ReLU, net_arch=[dict(pi=pi_hidden_units, vf=vf_hidden_units)]
)
tb_log_name = f"MultiReach_PPO_{learning_rate}" 
model_save_name = f"MultiReach_PPO_{learning_rate}.zip"
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
model.learn(
    total_timesteps=total_timesteps,
    tb_log_name=tb_log_name
)


model.save(model_save_name)