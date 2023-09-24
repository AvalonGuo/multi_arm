import numpy as np
from typing import Optional
import math
import pybullet as p
from multi_arm.pybullet import PyBullet
from multi_arm.envs.core import RobotTaskEnv
from multi_arm.envs.robots.multi_panda import Panda
from multi_arm.envs.tasks.inseration import Inseration

import torch as th 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class InserationPPOEnv(RobotTaskEnv):
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "dense",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode,renderer=renderer)
        robot = Panda(sim, body_name='object',block_gripper=True, base_position=np.array([-0.5, -0.04, 0.0]), control_type=control_type)
        #sim.set_base_pose(body="object",position=np.array([-0.5, 0.01, 0.0]),orientation=p.getQuaternionFromEuler([0,0,-math.pi/2]))
        task = Inseration(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

env = InserationPPOEnv(render_mode="human")

model = PPO.load(
    'PPO_Reach_0.0005',
    print_system_info=True,
    device="auto",
)

obs = env.predict_reset()
while 1:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Rendering in real-time (1x)
    #images.append(env.render())
    if terminated:
        obs = env.predict_reset()
        #images.append(env.render())