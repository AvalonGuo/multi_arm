import numpy as np
from typing import Optional
import math

from multi_arm.pybullet import PyBullet
from multi_arm.envs.pid_core import RobotTaskEnv
from multi_arm.envs.robots.multi_panda import Panda
from multi_arm.envs.tasks.inseration import Inseration

class InserationEnv(RobotTaskEnv):
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, body_name='object',block_gripper=False, base_position=np.array([-0.35, 0.0, 0.0]), control_type=control_type)
        
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

class PID:
    def __init__(self,P=0.2,I=0.0,D=0.0,desired_goal=0,id=0) -> None:
        self.id = id
        self.kp = P
        self.ki = I
        self.kd = D
        self.uPrevious = 0
        self.uCurrent = 0
        self.setValue = desired_goal
        self.lastErr = 0
        self.preLastErr = 0
        self.errSum = 0
        self.errSumLimit = 10
    
    def pidPosition(self,curValue):
        err = self.setValue - curValue
        dErr = err-self.lastErr
        self.preLastErr = self.lastErr
        self.lastErr = err
        self.errSum += err
        outPid = self.kp*err + (self.ki*self.errSum) + (self.kd * dErr)
        return outPid

    def pidIncrease(self,curValue):
        self.uCurrent = self.pidPosition(curValue)
        outPid  = self.uCurrent-self.uPrevious
        self.uPrevious = self.uCurrent
        return outPid

class Becontrolled:
    def __init__(self,id):
        self.id = id
        self.lastControlIn = 0
        self.preLastControlIn = 0
        self.lastControlOut = 0
        self.preLastControlOut = 0
        # 被控对象的相关计算
    def beControlledDeal(self, outPID):
        # output = 2*self.lastControlOut - 1*self.preLastControlOut + \
        #     0.00005*self.lastControlIn + 0.00005*self.preLastControlIn

        # output为被控对象的输出，此处是被控对象的传递函数离散化后，写成差分方程后的形式，被控对象的方程此处直接采用了设计好的参数，并与PID控制器的输出进行计算。
        # 如果需要设计自己的被控对象，将传递函数进行z变换后，交叉相乘，再进行z反变换即可，可参考《计算机控制系统》等书籍。
        # 因为是单位反馈，所以被控对象的输出等于传递函数的输入。
        output = 0.00019346*self.preLastControlIn + 0.00019671e-04*self.lastControlIn + \
            1.9512*self.lastControlOut - 0.9512*self.preLastControlOut
        self.preLastControlIn = self.lastControlIn
        self.lastControlIn = outPID
        self.preLastControlOut = self.lastControlOut
        self.lastControlOut = output
        return output

def pidControl(desired_goal:np.array,achieved_goal:np.array):
    #set_X,set_Y,set_Z = desired_goal[0],desired_goal[1],desired_goal[2]
    for i in range(4):
        pids[i].setValue = desired_goal[i]
        outPID = pids[i].pidIncrease(achieved_goal[i])
        achieved_goal[i] = becontrolleds[i].beControlledDeal(outPID)
    #    pid.setValue = desired_goal[i]
    #    outPID = pid.pidIncrease(achieved_goal[i])
    #    achieved_goal[i] = becontrolled.beControlledDeal(outPID)
    return achieved_goal

def is_success(desired_goal:np.array,achieved_goal:np.array):
    for i in range(desired_goal.shape[0]):
        if(abs(desired_goal[i]-achieved_goal[i])<=threshold):
            return True


def change_goal(desired_goal:np.array,state):
    if state == 0:
        desired_goal[3]+=0.05
    if state == 1:
        desired_goal[2]-=0.14
    if state == 2:
        desired_goal[3]-= 0.024
    if state == 3:
        desired_goal[2] += 0.1 
    if state == 4:
        desired_goal[0]=-0.032
        desired_goal[1]=0
    if state == 5:
        desired_goal[2] -= 0.07
    if state == 6:
        desired_goal[3] += 0.024
control_t = 0
c_t = 1/240.
duration_t = [2,8,8,4,4,8,4,1000000]
state = 0
pids = [PID(P=0,I=0.4,D=0,id=i) for i in range(4)]
becontrolleds = [Becontrolled(id=i) for i in range(4)]
env = InserationEnv(render_mode="human")
threshold = 0.001
desired_goal = np.array([0.2,0.05,0.20,0])
achieved_goal = np.array([0.0,0.0,0.0,0])
observation, info = env.reset()
#images = [env.render()]
i = 0
while 1:
    
    #action = env.action_space.sample() # random action
    #action = np.zeros(4)  #not_action
    #action = np.ones(4)
    action = pidControl(desired_goal=desired_goal,achieved_goal=achieved_goal)

    #print("type:",type(action),"shape:",action.shape)

    observation, reward, terminated, truncated, info = env.step(action)
    
    control_t += c_t

    if is_success(desired_goal=desired_goal,achieved_goal=observation["achieved_goal"])  and control_t>=duration_t[state]:
        control_t = 0
        change_goal(desired_goal,state=state)
        state += 1
    #images.append(env.render())
    if terminated or truncated:
        observation, info = env.reset()
        #images.append(env.render())

env.close()


from numpngw import write_apng

#write_apng("inseration.png", images, delay=40)  # real-time rendering = 40 ms between frames