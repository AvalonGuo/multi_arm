import numpy as np
from typing import Optional
import pybullet as p

import math
from multi_arm.pybullet import PyBullet
from multi_arm.envs.my_core import MyRobotTaskEnv
from multi_arm.envs.robots.panda import Panda
from multi_arm.envs.tasks.test import Grasp

import torch as th 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class GraspPPOEnv(MyRobotTaskEnv):
    def __init__(
        self,
        reward_type: str = "dense",
        control_type: str = "ee",
    ) -> None:
        self.sim = PyBullet(render=True)
        robot = Panda(body_name="panda",sim=self.sim,block_gripper=False, base_position=np.array([-0.5, -0.5, 0.0]), control_type=control_type)
        robot1 = Panda(body_name="panda2",sim=self.sim,block_gripper=False, base_position=np.array([0.5, -0.5, 0.0]), control_type=control_type)
        robot2 = Panda(body_name="panda3",sim=self.sim,block_gripper=False, base_position=np.array([0.5, 0.5, 0.0]), control_type=control_type)
        robot3 = Panda(body_name="panda4",sim=self.sim,block_gripper=False, base_position=np.array([-0.5, 0.5, 0.0]), control_type=control_type)
        self.sim.set_base_pose(body="panda",position= np.array([-0.5, -0.5, 0.0]),orientation= np.array([0,0,math.pi/2]))
        self.sim.set_base_pose(body="panda2",position= np.array([0.5, -0.5, 0.0]),orientation= np.array([0,0,math.pi/2]))
        self.sim.set_base_pose(body="panda3",position= np.array([0.5, 0.5, 0.0]),orientation= np.array([0,0,-math.pi/2]))
        self.sim.set_base_pose(body="panda4",position= np.array([-0.5, 0.5, 0.0]),orientation= np.array([0,0,-math.pi/2]))
        robots = [robot,robot1,robot2,robot3]
        self.robots = robots
        task = Grasp(self.sim,get_ee_position=robot.get_ee_position ,gripper_width=robot.get_fingers_width,reward_type=reward_type)

        super().__init__(
            robots,
            task
        )

env = GraspPPOEnv()



flag = 0
while 1:
    action1 = env.sim.get_base_position("object4")
    action2 = env.sim.get_base_position("object")
    action2[0]-=0.01
    #action = np.array([0.2,0.35,0.04,0.2])
    env.robots[0].pid_action(action1)
    env.robots[3].pid_action(action2)
    #env.robots[2].pid_action(np.array([0.55,0.3,0.14,0.005]))
    env.robots[2].pid_action(np.array([0.3,0.15,0.145]))
    #env.robots[2].close_gripper()
    #print(env.robots[2].get_ee_position())
    #print("self.sim.get:",env.sim.get_base_position("object2"))
    #env.robots[1].pid_action(np.array([0.45,-0.3,0.16,0.04]))
    if flag <= 10:
        env.sim.getContactPoints(env.robots[2].bodyID,env.robots[1].bodyID)
        flag+=1
    #env.robots[2].pid_action(np.array([0.67,-0.5,0.1,0.015]))
    env.sim.step()