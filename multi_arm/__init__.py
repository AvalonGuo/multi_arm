import os

from gymnasium.envs.registration import register

register(
    id="multi_arm/PandaMt-v3",
    entry_point="multi_arm.envs:PandaMtEnv",
)