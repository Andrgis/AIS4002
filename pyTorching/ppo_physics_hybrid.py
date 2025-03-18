import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pendulum_physics import CustomCartPoleEnv
import numpy as np

# Import your custom environment (ensure its registration code has been executed)
# For example, if it's defined in PPO_example_gpt.py, make sure to import that module.
from PPO_example_gpt import CustomCartPoleEnv

# Create the environment with continuous actions if needed.
env = gym.make('CustomCartPole-v1', render_mode=None)  # set render_mode="human" during evaluation only

# Wrap the environment if necessary.
env = DummyVecEnv([lambda: env])

# Define the PPO model.
# Use 'MlpPolicy' for a multi-layer perceptron (you may later experiment with different architectures).
model = PPO('MlpPolicy', env, verbose=1)

# Set total timesteps.
total_timesteps = 300000  # adjust as needed

# Train the agent.
model.learn(total_timesteps=total_timesteps)

# Save the trained model.
model.save(f"ppo_agents/ppo_cartpole_pure_rcos_wp_ts{total_timesteps//1000}k")

print("Training complete, model saved!")

