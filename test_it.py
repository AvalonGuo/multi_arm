from multi_arm.pybullet import PyBullet
from multi_arm.envs.tasks.multi_reach import Multi_Reach
import time
sim = PyBullet(render_mode="human")
task = Multi_Reach(sim)

while 1:
    task.reset()
    sim.step()