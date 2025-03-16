import gymnasium as gym
import numpy as np
from time import sleep
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pendulum_physics import CustomCartPoleEnv

# Load the trained model
model = PPO.load("ppo_agents/ppo_cartpole_hybrid_ts40k.zip")

# Test the trained model
env = gym.make('CustomCartPole-v1', render_mode="human")
env = DummyVecEnv([lambda: env])
obs = env.reset()
total_reward = 0
for _ in range(400):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    total_reward += rewards[0]
    #print(rewards)
    sleep(0.05)
print("Episode finished with total reward:", total_reward)

env.close()