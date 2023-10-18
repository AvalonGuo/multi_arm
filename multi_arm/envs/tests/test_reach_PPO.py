from typing import Optional
import numpy as np

from multi_arm.envs.tasks.reach import Reach
from multi_arm.envs.core import RobotTaskEnv
from multi_arm.pybullet import PyBullet
from multi_arm.envs.robots.panda import Panda

import torch as th 
from stable_baselines3 import PPO
class PandaReachEnv(RobotTaskEnv):
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
        task = Reach(sim=sim,reward_type=reward_type,get_ee_position=robot.get_ee_position)
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
env = PandaReachEnv(render_mode="rgb_array")
images = [env.render()]
model = PPO.load(
    'Reach_PPO_0.001',
    print_system_info=True,
    device="auto",
)

#############################
# Reset environment and get first observation
obs = env.predict_reset()

#############################
# Run trained agent in environment
for i in range(500):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()  # Rendering in real-time (1x)
    images.append(env.render())
    if done:
        obs = env.predict_reset()
        images.append(env.render())

from numpngw import write_apng

write_apng("test_reach.png", images, delay=40)  # real-time rendering = 40 ms between frames