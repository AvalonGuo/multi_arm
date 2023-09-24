from typing import Any, Dict, Union

import numpy as np

from multi_arm.envs.core import Task
from multi_arm.utils import distance


class ReachTo(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.2,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.get_ee_position = get_ee_position
        self.distance_threshold = distance_threshold
        self.object_size = 0.06
        self.goal_range_low = np.array([0, -goal_range / 2, 0.015])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, 0.015])
        self.state =0
        self.target2 = np.zeros(3)
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.4, width=0.7, height=0.4, x_offset=-0.1)
        self.sim.create_box(
            body_name="target1",
            mass=0.0,
            half_extents=np.array([0.015,0.015,0.015]),
            ghost=True,
            position=np.array([0,0,0.03]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1]),
        )
        self.sim.create_box(
            body_name="target2",
            mass=0.0,
            half_extents=np.array([0.015,0.015,0.015]),
            ghost=True,
            position=np.array([0,0.2,0.13]),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3])
        )
    def get_obs(self) -> np.ndarray:
        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:

        self.state = 0
        self.goal = self._sample_goal()
        
        self.sim.set_base_pose("target1", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.target2 = self.sim.get_base_position("target2")



    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.state==0 and d<self.distance_threshold:
            self.state=1
            self.goal = self.target2

            
        ##why reset not work
        #print("calculate",np.array(d < self.distance_threshold and self.state==1, dtype=bool))
        return np.array(d < self.distance_threshold and self.state==1, dtype=bool)


    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        #print("self.goal:",self.goal)
        d = distance(achieved_goal, desired_goal)
        target1_position = self.sim.get_base_position("target1")
        target2_position = self.sim.get_base_position("target2")
        d1 = distance(target1_position,target2_position)
        #if self.state ==0:
        #    d+=d1
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d
