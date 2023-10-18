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
        self.sim = PyBullet(render=True)
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
model = PPO.load(
    'MultiReach_PPO_0.001',
    print_system_info=True,
    device="auto",
)
obs = env.reset()
for i in range(5000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")  # Rendering in real-time (1x)
    if done:
        obs = env.reset()