# import pybullet as p
# import pybullet_data
# import numpy as np
# import time
# from stable_baselines3 import PPO

# # Set up PyBullet simulation environment
# p.connect(p.GUI)  # Connect to PyBullet GUI
# p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For loading URDF models
# p.setGravity(0, 0, -9.8)

# # Load URDF models
# plane_id = p.loadURDF("plane.urdf")
# spider_id = p.loadURDF("spider.urdf", [1, 1, 2])

# # Load a button-like object (can use a box as a simple button for demonstration)
# button_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05])
# button_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05], rgbaColor=[1, 0, 0, 1])
# button_start_pos = [0, 0, 0.1]  # Just above the plane
# button_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
# button_id = p.createMultiBody(baseMass=1, 
#                               baseCollisionShapeIndex=button_collision_shape, 
#                               baseVisualShapeIndex=button_visual_shape, 
#                               basePosition=button_start_pos, 
#                               baseOrientation=button_start_orientation)

# # Load the trained PPO model
# model = PPO.load("ppo_spider_model.zip")

# # Define action space and observation space dimensions
# num_joints = 6  # Number of joint positions your model expects
# observation_dim = 9  # Number of observations your model expects

# # Initialize the environment
# def get_observation():
#     leg_pos = []
#     for leg_idx in range(3):
#         pos = p.getLinkState(spider_id, leg_idx)[4]
#         leg_pos.append(pos)
#     return np.array(leg_pos).flatten()

# obs = get_observation()

# # Simulation loop
# for step in range(500):
#     # Predict action from the model
#     action, _states = model.predict(obs, deterministic=True)
    
#     # Apply action to the spider
#     scaled_action = np.interp(action, [-1, 1], [-np.pi/4, np.pi/4])
#     joint_indices = [0, 1, 2, 3, 4, 5]
#     p.setJointMotorControlArray(spider_id, joint_indices, p.POSITION_CONTROL, targetPositions=scaled_action)
    
#     # Step the simulation
#     p.stepSimulation()
    
#     # Update the camera view
#     focus_position, _ = p.getBasePositionAndOrientation(spider_id)
#     p.resetDebugVisualizerCamera(3, 30, -40, focus_position)
    
#     # Get the new observation
#     obs = get_observation()
    
#     # Optional: Check if the button is pressed (you can modify this based on your goal)
#     # You may want to include conditions to stop the simulation if a goal is reached

#     time.sleep(0.01)  # Slow down the simulation for better visualization

# # Close the simulation
# p.disconnect()




import pybullet as p
import pybullet_data
import numpy as np
import time
from stable_baselines3 import PPO

# Load the trained PPO model
model = PPO.load("ppo_spider_model.zip")

# Set up PyBullet simulation environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load URDF models
plane_id = p.loadURDF("plane.urdf")
spider_id = p.loadURDF("spider.urdf", [30, 10, 1])

# Initialize environment
def get_observation():
    leg_pos = []
    for leg_idx in range(3):
        pos = p.getLinkState(spider_id, leg_idx)[4]
        leg_pos.append(pos)
    return np.array(leg_pos).flatten()

obs = get_observation()

# Simulation loop
for step in range(500):
    # Predict action from the model
    action, _states = model.predict(obs, deterministic=True)

    # Scale action and apply it
    scaled_action = np.interp(action, [-1, 1], [-np.pi/2, np.pi/2])
    joint_indices = [0, 1, 2, 3, 4, 5]
    p.setJointMotorControlArray(spider_id, joint_indices, p.POSITION_CONTROL, targetPositions=scaled_action)

    # Step the simulation
    p.stepSimulation()

    # Update camera view
    focus_position, _ = p.getBasePositionAndOrientation(spider_id)
    p.resetDebugVisualizerCamera(3, 30, -40, focus_position)

    # Update observation
    obs = get_observation()

    time.sleep(0.01)

p.disconnect()




# import pybullet as p
# import pybullet_data
# import numpy as np
# import time
# from stable_baselines3 import PPO

# # Load the trained PPO model
# model = PPO.load("ppo_spider_model.zip")

# # Set up PyBullet simulation environment
# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.8)

# # Load URDF models
# plane_id = p.loadURDF("plane.urdf")
# spider_id = p.loadURDF("spider.urdf", [2, 0, 1])

# # Initialize environment
# def get_observation():
#     leg_pos = []
#     for leg_idx in range(3):
#         # Use the correct link indices based on jointInfo
#         hip_joint_index = 7 + leg_idx * 2  # hip_joint_leg1: 7, hip_joint_leg2: 9, hip_joint_leg3: 11
#         link_index = hip_joint_index
#         link_state = p.getLinkState(spider_id, link_index)
        
#         if link_state is None:
#             print(f"Warning: No link state found for link index {link_index}")
#             return np.zeros(9)  # Return a default state or handle appropriately
        
#         pos = link_state[4]
#         leg_pos.append(pos)
    
#     return np.array(leg_pos).flatten()

# obs = get_observation()

# # Simulation loop
# for step in range(500):
#     # Predict action from the model
#     action, _states = model.predict(obs, deterministic=True)
    
#     # Print action for debugging
#     print("Action:", action)
    
#     # Scale action and apply it
#     scaled_action = np.interp(action, [-1, 1], [-np.pi/2, np.pi/2])
#     joint_indices = [7, 8, 9, 10, 11, 12]  # Indices for all joints
#     p.setJointMotorControlArray(spider_id, joint_indices, p.POSITION_CONTROL, targetPositions=scaled_action)
    
#     # Step the simulation
#     p.stepSimulation()

#     # Update camera view
#     focus_position, _ = p.getBasePositionAndOrientation(spider_id)
#     p.resetDebugVisualizerCamera(3, 30, -40, focus_position)
    
#     # Update observation
#     obs = get_observation()
    
#     time.sleep(0.01)

# p.disconnect()

