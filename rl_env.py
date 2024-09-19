import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)    ###.DIRECT for non-GUI
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)

#load assets
p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
robot = p.loadURDF("spider.urdf",[0,0,1],[0,0,0,1])
object_of_focus=robot

# print(p.getnumJoints(robot))
jointid=4
jtype=p.getJointInfo(robot,jointid)[2]
jlowerlimit=p.getJointInfo(robot,jointid)[8]
jupperlimit=p.getJointInfo(robot,jointid)[9]

#set up camera
for step in range(300):
    joint_two_targ=np.random.uniform(jlowerlimit,jupperlimit)
    joint_four_targ=np.random.uniform(jlowerlimit,jupperlimit)
    p.setJointMotorControl2(robot,1,p.POSITION_CONTROL,targetPosition=joint_two_targ,force=500)
    p.setJointMotorControl2(robot,2,p.POSITION_CONTROL,targetPosition=joint_four_targ,force=500)
    focus_position,_=p.getBasePositionAndOrientation(robot)
    p.resetDebugVisualizerCamera(3, 30, -40, focus_position)
    p.stepSimulation()
    time.sleep(0.01)


# class SpiderEnv(gym.Env):
#     def __init__(self):
#         super(SpiderEnv, self).__init__()

#         # Action and Observation spaces
#         # Define the action space and observation space here
#         self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # Example action space
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)  # Example observation space

#         # Connect to PyBullet and load the environment
#         p.connect(p.GUI)
#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         p.setGravity(0, 0, -9.8)
#         self.plane_id = p.loadURDF("plane.urdf")
#         self.spider_id = p.loadURDF("spider.urdf", [1, 1, 2])

#         # Initialize other environment variables here

#     def step(self, action):
#         # Apply actions to the environment
#         # Example: Set joint torques or forces
#         # action = np.clip(action, self.action_space.low, self.action_space.high)
#         # p.setJointMotorControlArray(self.spider_id, jointIndices, controlMode=p.TORQUE_CONTROL, forces=action)
        
#         # Step the simulation
#         p.stepSimulation()
        
#         # Get observations
#         # Example: Read joint positions, velocities, etc.
#         # observation = ...

#         # Compute reward
#         reward = self.get_reward()

#         # Check if episode is done
#         done = False  # Define your own condition

#         return observation, reward, done, {}

#     def reset(self):
#         # Reset the environment state
#         p.resetSimulation()
#         p.setGravity(0, 0, -9.8)
#         self.plane_id = p.loadURDF("plane.urdf")
#         self.spider_id = p.loadURDF("spider.urdf", [1, 1, 2])
#         # Reset other variables
        
#         # Return initial observations
#         # Example: observation = ...
#         return observation

#     def get_reward(self):
#         # Implement the reward function based on the PressWithSpecificForce task
#         # Example: reward = ...
#         return reward

#     def render(self, mode='human'):
#         pass  # Optional: Implement if you want to visualize the environment

#     def close(self):
#         p.disconnect()
