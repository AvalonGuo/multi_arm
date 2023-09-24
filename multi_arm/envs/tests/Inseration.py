import numpy as np
from typing import Optional
import pybullet as p
import math
from multi_arm.pybullet import PyBullet
from multi_arm.envs.core import RobotTaskEnv
from multi_arm.envs.robots.panda import Panda
from multi_arm.envs.tasks.reachTo import ReachTo
from multi_arm.envs.tasks.Grasp import Grasp

import torch as th 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env



class InserationPPOEnv(RobotTaskEnv):
    def __init__(
        self,
        reward_type: str = "dense",
        control_type: str = "ee",
    ) -> None:
        sim = PyBullet(render=True)
        robot = Panda(sim,body_name="panda",block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        gripper_width = robot.get_fingers_width
        task = Grasp(sim,robot=robot,reward_type=reward_type)
        super().__init__(
            robot,
            task
        )

env = InserationPPOEnv()
env.reset()

while 1:

    action = env.action_space.sample()
    env.step(action)

    #for i in range(10):
    #    print(i)
    #env.reset()