import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC

class ArmEnv(gym.Env):
    def __init__(self):
        super(ArmEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.step_count = 0
        self.state = None
        self.spiderUID = None
        self.init_state()

    def init_state(self):
        p.connect(p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        p.setGravity(0, 0, -9.8)

        self.spiderUID = p.loadURDF("spider.urdf", [30, 10, 1], [0, 0, 0, 1])
        p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])

        leg_pos = []
        for leg_idx in range(3):
            pos = p.getLinkState(self.spiderUID, leg_idx)[4]
            leg_pos.append(pos)

        self.state = np.array(leg_pos).flatten()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        self.init_state()
        self.step_count = 0
        return self.state, {}  # Return state and an empty info dict
        

    def step(self, action):
        self.step_count += 1

        # Scale action from [-1, 1] to a more appropriate range for joint positions
        scaled_action = np.interp(action, [-1, 1], [-np.pi/2, np.pi/2])

        joint_indices = [0, 1, 2, 3, 4, 5]
        p.setJointMotorControlArray(self.spiderUID, joint_indices, p.POSITION_CONTROL, targetPositions=scaled_action)

        p.stepSimulation()

        leg_pos = []
        for leg_idx in range(3):
            pos = p.getLinkState(self.spiderUID, leg_idx)[4]
            leg_pos.append(pos)

        self.state = np.array(leg_pos).flatten()

        # Calculate distance from the origin (button position)
        distance_from_button = np.linalg.norm(self.state)

        # Reward is inversely proportional to the distance, with a reward of 100 for reaching the origin
        reward = 100 if distance_from_button < 0.1 else -distance_from_button

        terminated = self.step_count >= 500
        truncated = False

        return self.state, reward, terminated, truncated, {}


    def render(self):
        pass  # Implement if you want to visualize the environment

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    # Create a vectorized environment
    env = DummyVecEnv([lambda: ArmEnv()])

    # Initialize and train PPO model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000000)

    m2 = SAC("MlpPolicy", env, verbose=1)
    m2.learn(total_timesteps=10000000)

    # Save the trained model
    model.save("ppo_spider_model")
    m2.save("sac_spider_model")

    # Load the model and test it


    # model = PPO.load("ppo_spider_model")
    
    # # Create a new environment for testing
    # test_env = ArmEnv()
    # obs, _ = test_env.reset()
    
    # # Switch to GUI mode for visualization
    # p.disconnect()
    # p.connect(p.GUI)
    # test_env.init_state()

    # for _ in range(500):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = test_env.step(action)
        
    #     # Update camera
    #     focus_position, _ = p.getBasePositionAndOrientation(test_env.spiderUID)
    #     p.resetDebugVisualizerCamera(3, 30, -40, focus_position)
        
    #     if terminated or truncated:
    #         obs, _ = test_env.reset()

    # test_env.close()