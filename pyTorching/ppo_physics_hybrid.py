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
total_timesteps = 50000  # adjust as needed

# Train the agent.
model.learn(total_timesteps=total_timesteps)

# Save the trained model.
model.save(f"ppo_agents/ppo_cartpole_hybrid_ts{total_timesteps//1000}k")

print("Training complete, model saved!")


'''import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create the CartPole environment
env = gym.make('Pendulum-v1')
env = DummyVecEnv([lambda: env])

# Define the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=300000)

# Save the model
model.save(f"ppo_agents/ppo_pendulum_hybrid_ts300k")

# Load the trained model
model = PPO.load("ppo_agents/ppo_pendulum_hybrid_test3")

# Test the trained model
env = gym.make('Pendulum-v1', render_mode="human")
env = DummyVecEnv([lambda: env])
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")

env.close()'''