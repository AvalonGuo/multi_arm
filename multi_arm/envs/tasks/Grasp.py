from typing import Any, Dict,Union
import numpy as np

from multi_arm.envs.core import Task
from multi_arm.utils import distance
from multi_arm.pybullet import PyBullet
from multi_arm.envs.robots.panda import Panda
class Grasp(Task):
    def __init__(
        self, 
        sim: PyBullet,
        robot: Panda,
        reward_type: str = "sparse",
        distance_threshold: float = 0.02,
        goal_range: float = 0.2, 
        ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.robot = robot
        self.get_ee_position = robot.get_ee_position
        self.gripper_width = robot.get_fingers_width
        self.close_gripper = robot.close_gripper
        self.object_size = 0.05
        self.success_num = 0
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0.0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, 0.0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-50)


    def _create_scene(self) -> None:
        """Create the scene."""
        #self.sim.create_plane(z_offset=-0.4)
        #self.sim.create_table(length=20, width=20, height=0.4, x_offset=-0.1)  #origin 1.4 0.7

        
        self.sim.create_box(
            body_name="object",
            half_extents=np.array([0.015,0.015,0.015]),
            mass=35.,
            position=np.array([0,0,0.015]),
            rgba_color=np.array([0.1,1,0.1,1]),
            lateral_friction=4,
            spinning_friction=4,
        )


        #self.sim.create_box(
        #    body_name="target",
        #    half_extents=np.array([0.015,0.015,0.015]),
        #    mass=0.,
        #    position=np.array([0,0,0.065]),
        #    rgba_color=np.array([1.0,0.1,0.1,0.3]),
        #    ghost=True
        #)
        self.sim.loadURDF(
            body_name="plane",
            fileName = "plane.urdf"
        )


    def get_obs(self) -> np.ndarray:
        self.goal = self.sim.get_base_position("object") # Need to update goal as it changes
        
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        # object_rotation = self.sim.get_base_rotation("object")
        # object_velocity = self.sim.get_base_velocity("object")
        # object_angular_velocity = self.sim.get_base_angular_velocity("object")
        # observation = np.concatenate([object_position, object_rotation, object_position_2]) 
        # observation = np.concatenate([object_position, object_rotation])  # object_velocity, object_angular_velocity])
        # observation = 
        return object_position

    def get_achieved_goal(self) -> np.ndarray:
        #object_position = np.array(self.sim.get_base_position("object"))
        # return object_position
        desired_goal = self.sim.get_base_position("object")
        desired_goal[2]+=0.01 
        ee_position = np.array(self.get_ee_position())
        d = distance(ee_position,desired_goal)
        if(d<self.distance_threshold):
            self.close_gripper()
        #fingers_width = self.gripper_width()
        #ee_position = np.concatenate((ee_position, [fingers_width])) #HERE
        return ee_position

    def reset(self) -> None:
        # self.goal = self._sample_goal()
        # NOTE: testing change here since not pick and place
        object_position = self._sample_object()
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.robot.reset() 
        self.goal = self.sim.get_base_position("object")
        
        self.success_num +=1
        print("success_num:",self.success_num)
        # self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
 


    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, 0.015,0])  # z offset for the cube center        #HERE
        #noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        # if self.np_random.random() < 0.3:
        #     noise[2] = 0.0
        #goal[0:3] += noise
        return goal[0:3]

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0,0.015])
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        
        min_gripper_width = 0.005
        max_gripper_width = 0.032
        achieved_goal = self.get_achieved_goal()
        desired_goal=desired_goal.copy()
        desired_goal[2]+=0.01
        gripper_width = self.gripper_width()
        ## Better Approach
       
        d = distance(achieved_goal, desired_goal)
        # achieved_goal[3] > 0.1


        return np.array(((d < self.distance_threshold) and(gripper_width>min_gripper_width)  and (gripper_width<max_gripper_width)), dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        desired_goal = self.sim.get_base_position("object")
        desired_goal[2]+=0.01
        achieved_goal = self.get_achieved_goal() # Better Approach
        d = distance(achieved_goal, desired_goal)
        # Note: Could add a gripper based reward here


        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d