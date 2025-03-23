import gymnasium as gym
import numpy as np
from time import sleep
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pendulum_physics import CustomCartPoleEnv

# Load the trained model
model = PPO.load("ppo_agents/ppo_cartpole_allgood_ts505k.zip")

# Test the trained model
env = gym.make('CustomCartPole-v1', render_mode="human")
env = DummyVecEnv([lambda: env])
obs = env.reset()
total_reward = 0
theta_vec = []
theta_dot_vec = []
x_vec = []
x_dot_vec = []
for _ in range(400):
    action, _states = model.predict(obs)
    theta = obs[0][2]
    '''if theta > 0.05:
        theta -= 2 * np.pi'''
    theta_vec.append(theta)
    theta_dot_vec.append(obs[0][3])
    x_vec.append(obs[0][0])
    x_dot_vec.append(obs[0][1])
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    total_reward += rewards[0]
    print(rewards)
print("Episode finished with total reward:", total_reward)


env.close()

# Plot the state vectors over time
import matplotlib.pyplot as plt

fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
t = np.arange(0, 8, 0.02)

axs[0].plot(t, theta_vec, label="Theta", color='b')
axs[0].set_ylabel("Theta (rad)")
axs[0].grid()
axs[0].legend()

axs[1].plot(t, theta_dot_vec, label="Theta_dot", color='g')
axs[1].set_ylabel("Theta_dot (rad/s)")
axs[1].grid()
axs[1].legend()

axs[2].plot(t, x_vec, label="X", color='r')
axs[2].set_ylabel("X")
axs[2].grid()
axs[2].legend()

axs[3].plot(t, x_dot_vec, label="X_dot", color='m')
axs[3].set_ylabel("X_dot")
axs[3].set_xlabel("Time (s)")
axs[3].grid()
axs[3].legend()

fig.suptitle("State vectors over time")
plt.show()