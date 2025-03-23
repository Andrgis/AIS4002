import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv  # base class for convenience
import numpy as np
import math
from gymnasium import spaces

# Physical parameters (from your simulation)
m_p = 0.1       # Pole mass [kg]
m_c = 0.1       # Cart mass [kg] (assumed to be the same as pole mass for simplicity)
l = 0.095       # Pole length [m]
l_com = l / 2   # Distance to center of mass of pole
J = (1/3) * m_p * l * l  # Inertia of the pole [kg*m^2]
g = 9.81        # Gravitational acceleration [m/s^2]
total_mass = m_p + m_c
polemass_length = m_p * l_com

# Control loop parameter for integration
dt = 0.02  # timestep (s)

# Global variables for swingup function (for simplicity, you may want to store these in the env)
prevAngle = 0.0

class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self, render_mode=None):
        # Call parent with render_mode (must be accepted)
        super().__init__(render_mode=render_mode)
        # You can adjust the cart’s position limits
        self.x_threshold = 5.0


        self.state = None

        # The state is [x, x_dot, theta, theta_dot]
        self.observation_space = spaces.Box(
            low=np.array([-self.x_threshold, -np.inf, -np.pi, -np.inf], dtype=np.float32),
            high=np.array([self.x_threshold, np.inf, np.pi, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        # Define a discreet action space (e.g., acceleration in m/s^2)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start with the pole hanging down (theta = pi)
        self.state = np.array([0.0, 0.0, 3.1, 0.0], dtype=np.float32)
        # Reset global swingup variable
        global prevAngle
        prevAngle = self.state[2]
        return self.state, {}

    def step(self, action):
        # Map discrete actions to acceleration values.
        acceleration_map = {0: -2.5, 1: 2.5}

        # Unpack current state
        x, x_dot, theta, theta_dot = self.state

        # Agent provides acceleration (action)
        u = acceleration_map[action]

        # Now, compute dynamics.
        # Using the equations similar to CartPole but replacing force with m_c * u.
        force = m_c * u  # force = mass_cart * acceleration

        # Using the dynamics from the original CartPole:
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
        thetaacc = (g * sintheta - costheta * temp) \
                   / (l * (4.0/3.0 - m_p * costheta**2 / total_mass)) \
                   - 0.05*theta_dot
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Update state using Euler integration
        x = x + dt * x_dot
        x_dot = x_dot + dt * xacc
        theta = theta + dt * theta_dot
        theta = (theta + np.pi) % (2 * np.pi) - np.pi  # Normalize angle to [-pi, pi]
        theta_dot = theta_dot + dt * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)  # Ensure float32

        # Termination: Only if cart goes out of bounds
        done = bool(x < -self.x_threshold or x > self.x_threshold)

        reward = (0.5 + 0.5 * np.cos(theta))**2 - 0.002 * theta_dot**2

        if done:
            reward -= 10

        return self.state, reward, done, False, {}

    def render(self):
        # You could use the original rendering from CartPoleEnv,
        # or implement your own. For simplicity, we call the parent's render.
        return super().render()

# Register the custom environment.
gym.envs.registration.register(
    id='CustomCartPole-v1',
    entry_point="pendulum_physics:CustomCartPoleEnv", #"__main__:CustomCartPoleEnv",
    max_episode_steps=400,
    reward_threshold=475.0,
)

# Example usage:
if __name__ == '__main__':
    env = gym.make("CustomCartPole-v1", render_mode="human")
    state, _ = env.reset()
    total_reward = 0
    done = False
    print(env.observation_space)
    while not done:
        # For testing, we can use a simple heuristic action (here, 0, not used in swingup mode)
        action = 0.0
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(state)
        env.render()
    env.close()
    print("Episode finished with total reward:", total_reward)