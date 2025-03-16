import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pendulum_physics import CustomCartPoleEnv

# Load the trained model
model = PPO.load("ppo_agents/ppo_cartpole_hybrid_ts50k.zip")

# Test the trained model
env = gym.make('CustomCartPole-v1', render_mode="human")
env = DummyVecEnv([lambda: env])
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")

env.close()