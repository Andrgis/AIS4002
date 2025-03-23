import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from pendulum_physics import CustomCartPoleEnv
import numpy as np

tmp_path = "./tmp/sb3_log/"  # Path to save logs
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


# Create the environment with continuous actions if needed.
env = Monitor(gym.make('CustomCartPole-v1', render_mode=None),
              filename=tmp_path)  # set render_mode="human" during evaluation only

# Wrap the environment if necessary.
env = DummyVecEnv([lambda: env])

# Define the PPO model.
# Use 'MlpPolicy' for a multi-layer perceptron (you may later experiment with different architectures).
model = PPO('MlpPolicy', env, verbose=1)
model.set_logger(new_logger)
# Set total timesteps.
total_timesteps = 350000  # adjust as needed

# Train the agent.
model.learn(total_timesteps=total_timesteps)

# Save the trained model.
model.save(f"ppo_agents/ppo_cartpole_allgood_ts{total_timesteps // 1000}k")

print("Training complete, model saved!")

