import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

class ArmEnv():
    def __init__(self):
        self.state = self.init_state()
        self.step_count = 0

    def init_state(self):
        # Initialize PyBullet in DIRECT mode (non-GUI)
        p.connect(p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        p.setGravity(0, 0, -9.8)

        # Load the spider URDF and a plane
        self.spiderUID = p.loadURDF("spider.urdf", [0, 0, 1], [0, 0, 0, 1])
        p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])

        # Get initial leg positions (tracking the position of all 3 legs)
        leg_pos = []
        for leg_idx in range(3):  # Assuming leg links are 0, 1, 2
            pos = p.getLinkState(self.spiderUID, leg_idx)[4]  # Get the world position of the end-effector (foot)
            leg_pos.append(pos)

        # Flatten to form an observation vector
        obs = np.array(leg_pos).flatten()
        return obs

    def reset(self):
        # Disconnect and reset the simulation
        p.disconnect()
        self.state = self.init_state()
        self.step_count = 0

    def step(self, new_pos):
        # Increment the step count
        self.step_count += 1

        # Update the positions of the joints (assuming the joints to be hip and knee for 3 legs)
        # Assuming joints [0, 1] for leg 1, [2, 3] for leg 2, and [4, 5] for leg 3
        # `new_pos` should be an array of 6 positions, one for each joint (hip and knee)
        joint_indices = [0, 1, 2, 3, 4, 5]  # Joint indices for 3 legs (hip and knee)
        p.setJointMotorControlArray(self.spiderUID, joint_indices, p.POSITION_CONTROL, targetPositions=new_pos)

        # Step the simulation
        p.stepSimulation()

        # Get the new positions of the legs' end-effectors
        leg_pos = []
        for leg_idx in range(3):  # Get the world position of the end-effectors (feet)
            pos = p.getLinkState(self.spiderUID, leg_idx)[4]
            leg_pos.append(pos)

        # Observation after the step (flattened position of all legs)
        obs = np.array(leg_pos).flatten()

        # Reset the environment if step limit is reached
        if self.step_count >= 50:
            self.reset()
            obs = np.array([p.getLinkState(self.spiderUID, leg_idx)[4] for leg_idx in range(3)]).flatten()
            reward = -1  # Fixed negative reward for ending the episode early
            done = True
            return obs, reward, done

        # Example reward (can be modified based on task)
        reward = -1  # Placeholder reward, modify based on your task (e.g., reaching a target position)
        done = False

        return obs, reward, done

def compute_reward(leg_pos, button_pos):
    distance = np.linalg.norm(leg_pos - button_pos)
    reward = -distance  # Negative reward for distance from the button
    if distance < 0.1:
        reward += 100  # Large reward for pressing the button
    return reward




if __name__ == "__main__":
    # Create a vectorized environment (required by Stable Baselines3)
    env = DummyVecEnv([lambda: ArmEnv()])

    # Initialize and train PPO model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)  # Train the model

    # Save the trained model
    model.save("ppo_spider_model")

    # Load the model and test it
    model = PPO.load("ppo_spider_model")
    obs = env.reset()
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()  # Optional for GUI rendering
        if done:
            obs = env.reset()





















# env = ArmEnv()
# obs = env.init_state()

# # Example usage with ArmEnv
# env = ArmEnv()

# # Define lower and upper joint limits for each of the 6 joints (hip and knee for each leg)
# jlowerlimit = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]  # Joint lower limits
# jupperlimit = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]       # Joint upper limits

# # Run the environment for 500 steps
# for step in range(500):
#     # Generate a random target position for each of the 6 joints within the limits
#     new_pos = np.random.uniform(jlowerlimit, jupperlimit)

#     # Step the environment with the generated action
#     obs, reward, done = env.step(new_pos)

#     # Print the current observation (positions of leg end-effectors)
#     print(f"Step {step}: Observation = {obs}, Reward = {reward}, Done = {done}")

#     # If the environment signals 'done', break the loop
#     if done:
#         break

#     # Step the simulation in PyBullet
#     p.stepSimulation()

