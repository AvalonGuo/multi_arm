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
        robot = Panda(sim,body_name="panda",block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = ReachTo(sim,get_ee_position=robot.get_ee_position ,reward_type=reward_type)
        super().__init__(
            robot,
            task
        )

env = ReachToPPOEnv()
model = PPO.load(
    'ReachTo_PPO_0.001',
    print_system_info=True,
    device="auto",
)

#############################
# Reset environment and get first observation
obs = env.reset()
images = [env.render(mode="rgb_array")]
#############################
# Run trained agent in environment
for i in range(100):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    images.append(env.render(mode="rgb_array"))

    if done:
        obs = env.reset()
        images.append(env.render(mode="rgb_array"))

env.close()


from numpngw import write_apng

write_apng("ReachTo_PPO.png", images, delay=40)  # real-time rendering = 40 ms between frames