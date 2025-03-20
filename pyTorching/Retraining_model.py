import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pendulum_physics import CustomCartPoleEnv
import numpy as np

# Create the environment
env = gym.make('CustomCartPole-v1', render_mode=None)
env = DummyVecEnv([lambda: env])

# Path to the saved model
modelnum = 400000  # Change as needed
model_path = f"ppo_agents/ppo_cartpole_pure_simple_ts{modelnum//1000}k.zip"  # Change to the actual saved file

# Try to load an existing model if available
try:
    model = PPO.load(model_path, env=env)  # Load existing model
    print("Loaded existing model:", model_path)
except FileNotFoundError:
    print("No saved model found, training from scratch.")
    model = PPO('MlpPolicy', env, verbose=1)  # Train a new model if not found

# Continue training with additional timesteps
additional_timesteps = 100000  # Change as needed
model.learn(total_timesteps=additional_timesteps)

# Save the updated model
new_model_path = f"ppo_agents/ppo_cartpole_pure_simple_ts{(modelnum + additional_timesteps) // 1000}k"
model.save(new_model_path)

print(f"Training complete, model saved to {new_model_path}!")
