import pybullet as p
import pybullet_data
import time

# Connect to the PyBullet GUI
p.connect(p.GUI)

# Set the additional search path for PyBullet to find the URDF files
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For loading URDF models

# Set gravity in the simulation
p.setGravity(0, 0, -9.8)

# Load a plane as the ground
plane_id = p.loadURDF("plane.urdf")

# Load the spider URDF model slightly above the plane (0.05 meters above to prevent embedding)
spider_id = p.loadURDF("spider.urdf", [1, 1, 2])  # Starting just above the plane

# Make sure collisions are enabled
p.changeDynamics(plane_id, -1, lateralFriction=1.0)  # Add friction to the plane
p.changeDynamics(spider_id, -1, lateralFriction=1.0)  # Add friction to the spider

# Start the simulation loop
for _ in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)  # Sleep to match real-time simulation

# Disconnect from the simulation
p.disconnect()
