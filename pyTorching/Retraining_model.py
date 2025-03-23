import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pendulum_physics import CustomCartPoleEnv
import numpy as np
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import matplotlib.pyplot as plt
import os

main_log = "./tmp/sb3_log_main/"  # Main log directory
adder_log = "./tmp/adder/"  # New log directory

# Ensure directories exist
os.makedirs(main_log, exist_ok=True)
os.makedirs(adder_log, exist_ok=True)

# Configure the logger
new_logger = configure(adder_log, ["stdout", "csv", "tensorboard"])

# Create and wrap the environment with Monitor
env = gym.make('CustomCartPole-v1', render_mode=None)
env = Monitor(env, filename=adder_log + "monitor.csv")  # Explicit CSV path
env = DummyVecEnv([lambda: env])

# Path to the saved model
modelnum = 350000  # Change as needed
model_path = f"ppo_agents/ppo_cartpole_allgood_ts{modelnum//1000}k.zip"  # Change as needed

# Try to load an existing model
try:
    model = PPO.load(model_path, env=env)
    print("Loaded existing model:", model_path)
except FileNotFoundError:
    raise FileNotFoundError("No saved model found")

# Set the new logger
model.set_logger(new_logger)

# Continue training
additional_timesteps = 155000
model.learn(total_timesteps=additional_timesteps)

# Save the updated model
new_model_path = f"ppo_agents/ppo_cartpole_allgood_ts{(modelnum + additional_timesteps) // 1000}k"
model.save(new_model_path)

print(f"Training complete, model saved to {new_model_path}!")

# Load existing CSV data (if available)
existing_csv_path = main_log + "monitor.csv"
new_csv_path = adder_log + "monitor.csv"

if os.path.exists(existing_csv_path):
    existing_data = pd.read_csv(existing_csv_path)
else:
    existing_data = pd.DataFrame()  # Create an empty DataFrame if no previous log exists

# Load new CSV data
if os.path.exists(new_csv_path):
    new_data = pd.read_csv(new_csv_path)
else:
    raise FileNotFoundError("New monitor log not found after training.")

# Append new data to existing data if columns match
if not existing_data.empty and list(existing_data.columns) != list(new_data.columns):
    raise ValueError("CSV column mismatch. Check log format consistency.")

combined_data = pd.concat([existing_data, new_data], ignore_index=True)

# Save combined data back to main log directory
combined_data.to_csv(existing_csv_path, index=False)

print(f"Logs merged and saved to {existing_csv_path}!")
