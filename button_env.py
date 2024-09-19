import pybullet as p
import pybullet_data
import numpy as np
import time

# Set up PyBullet simulation environment
p.connect(p.GUI)  # Connect to PyBullet GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For loading URDF models
p.setGravity(0, 0, -9.8)

# Load a simple plane as the ground
plane_id = p.loadURDF("plane.urdf")
spider_id = p.loadURDF("spider.urdf", [1, 1, 2]) 

# Load a button-like object (can use a box as a simple button for demonstration)
button_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05])
button_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05], rgbaColor=[1, 0, 0, 1])

# Button starting position and orientation
button_start_pos = [0, 0, 0.1]  # Just above the plane
button_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

# Create a button body
button_id = p.createMultiBody(baseMass=1, 
                              baseCollisionShapeIndex=button_collision_shape, 
                              baseVisualShapeIndex=button_visual_shape, 
                              basePosition=button_start_pos, 
                              baseOrientation=button_start_orientation)

# Button push depth and original position
button_push_depth = 0.05  # How much the button moves down when pushed
button_original_pos = np.array(button_start_pos)  # Keep track of the original position
button_pushed_pos = button_original_pos.copy()
button_pushed_pos[2] -= button_push_depth  # Pushed position

# Flag to know if button is currently pressed
is_button_pushed = False

# Function to push and release the button
def push_button():
    global is_button_pushed
    if not is_button_pushed:
        p.resetBasePositionAndOrientation(button_id, button_pushed_pos, button_start_orientation)
        is_button_pushed = True
    else:
        p.resetBasePositionAndOrientation(button_id, button_original_pos, button_start_orientation)
        is_button_pushed = False

for step in range(500):
    focus_position,_=p.getBasePositionAndOrientation(spider_id)
    p.resetDebugVisualizerCamera(3, 30, -40, focus_position)
    p.stepSimulation()
    time.sleep(0.01)