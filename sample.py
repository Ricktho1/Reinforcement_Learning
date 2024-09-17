import pybullet as p
import time

# Connect to the PyBullet GUI
p.connect(p.GUI)

# Load the URDF file
spider_id = p.loadURDF("spider.urdf")

# Set gravity to allow for the physics simulation
p.setGravity(0, 0, -9.8)

# Start simulation loop
for _ in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)  # Sleep to match real-time simulation

# Disconnect after the simulation
p.disconnect()
