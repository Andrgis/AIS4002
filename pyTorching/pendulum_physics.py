import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv  # base class for convenience
import numpy as np
import math

# Physical parameters (from your simulation)
m_p = 0.1       # Pole mass [kg]
m_c = 1.0       # Cart mass [kg] (you can adjust)
l = 0.095       # Pole length [m]
l_com = l / 2   # Distance to center of mass of pole
J = (1/3) * m_p * l * l  # Inertia of the pole [kg*m^2]
g = 9.81        # Gravitational acceleration [m/s^2]
total_mass = m_p + m_c
polemass_length = m_p * l_com

# Swingup parameters
Er = 0.015      # Reference energy (J)
ke = 10         # Gain for swingup
u_max = 2.5     # Maximum acceleration [m/s^2]
balance_range = 20.0  # Degrees: within this range, use balance mode

# Control loop parameter for integration
dt = 0.02  # timestep (s)

# Global variables for swingup function (for simplicity, you may want to store these in the env)
prevAngle = 0.0

def swingup_acceleration(angle: float) -> np.ndarray:
    """Compute acceleration for swingup mode based on the current angle."""
    global prevAngle
    angularV = (angle - prevAngle) / dt
    prevAngle = angle

    # Total energy of the pendulum
    E = 0.5 * J * angularV**2 + m_p * g * l_com * (1 - np.cos(angle))
    # Limit energy difference
    E_diff = np.clip(E - Er, -Er, Er)
    u = ke * E_diff * (-angularV * np.cos(angle))
    u_sat = np.clip(u, -u_max, u_max)
    print(u_sat)
    return u_sat

class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self, render_mode=None):
        # Call parent with render_mode (must be accepted)
        super().__init__(render_mode=render_mode)
        # Remove the angle termination (allow full swing)
        self.theta_threshold_radians = float('inf')
        # You can adjust the cart’s position limits if desired
        self.x_threshold = 5.0

        # We'll override dynamics, so we define our own state update.
        # The state is [x, x_dot, theta, theta_dot]
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start with the pole hanging down (theta = pi)
        self.state = np.array([0.0, 0.0, 0.8*np.pi, 0.0])
        # Reset global swingup variable
        global prevAngle
        prevAngle = self.state[2]
        return self.state, {}

    def step(self, action):
        """
        In this hybrid approach:
          - If the pole angle (in degrees) is outside the balance_range, use the physics-based swingup.
          - If within the balance_range, use the agent’s control input (assumed to be a continuous acceleration value).
        """
        # Unpack current state
        x, x_dot, theta, theta_dot = self.state

        # Convert pole angle to degrees for mode switching
        theta_deg = math.degrees((theta + math.pi) % (2*math.pi) - math.pi)

        # Determine control mode:
        if abs(theta_deg) > balance_range:
            # Swing-up mode: use physics-based swingup to get acceleration
            u = swingup_acceleration(theta)
        else:
            # Balance mode: agent provides acceleration (action)
            # Here we assume the agent's action is a scalar in [-1,1] and scale it to the physical limits.
            # You may choose to tune this scaling.
            u = np.clip(action, -1.0, 1.0) * u_max

        # Now, compute dynamics.
        # Instead of calling super().step, we integrate our own equations.
        # Using the equations similar to CartPole but replacing force with m_c * u.
        force = m_c * u  # force = mass_cart * acceleration

        # Using the dynamics from the original CartPole:
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
        thetaacc = (g * sintheta - costheta * temp) / (l * (4.0/3.0 - m_p * costheta**2 / total_mass)) - 0.2*theta_dot
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Update state using Euler integration
        x = x + dt * x_dot
        x_dot = x_dot + dt * xacc
        theta = theta + dt * theta_dot
        theta_dot = theta_dot + dt * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

        # Termination: Only if cart goes out of bounds
        done = bool(x < -self.x_threshold or x > self.x_threshold)
        # Define reward: you could reward based on how close the pole is to upright.
        # For example, reward = 1 when pole is perfectly upright (theta ~ 0).
        reward = (0.5 + 0.5 * np.cos(theta)) - 0.002 * theta_dot**2
        if done:
            reward -= 100

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
    while not done:
        # For testing, we can use a simple heuristic action (here, 0, not used in swingup mode)
        action = 0.0
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()
    env.close()
    print("Episode finished with total reward:", total_reward)



'''# from QUBE import QUBE
from time import sleep, time
import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Physical parameters
m_p = 0.1  # Pendulum stick mass (kg)
l = 0.095  # Length of pendulum (m)
l_com = l / 2  # Distance to center of mass (m)
J = (1 / 3) * m_p * l * l  # Inertia (kg*m^2)
g = 9.81  # Gravitational constant (m/s^2)

# Swingup parameters
Er = 0.015  # Reference energy (Joules)
ke = 10  # Tunable gain for swingup voltage (m/s/J)
u_max = 2.5  # Max voltage (m/s^2)
balance_range = 20.0  # Range where mode switches to balancing (degrees)

# Control loop parameters
freq = 300  # Frequency (Hz)
dt = 1.0 / freq  # Timestep (sec)

# Program variables
targetPos = 0
targetAngle = 0
t_balance = 0
prevAngle = 0
prevPos = 0
last = time()
t_reset = time()
mode = 0
lastMode = 0
reset = False


def swingup(angle: float) -> float:
    """
    :param angle: Takes in the angle and calculates the acceleration
    :return: u: Control output [m/s/s]
    """
    global prevAngle
    angularV = (angle - prevAngle) / dt
    print(angularV)
    prevAngle = angle

    # E = E_k + E_p
    E = 0.5 * J * angularV ** 2 + m_p * g * l_com * (1 - np.cos(angle))
    E_diff = max(-Er, min(E - Er, Er))
    u = ke * E_diff * (-angularV * np.cos(angle))
    # u = ke * (E - Er) * (-angularV * np.cos(angle))
    u_sat = max(-u_max, min(u, u_max))
    return u_sat


# Reinforcement learning for stability

# Create the CartPole environment
class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        #self.length = 0.7  # Modify pole length
        #self.masscart = 1.5  # Modify cart mass
        # Add more customizations as needed
        # Remove the angle threshold by setting it to infinity.
        self.theta_threshold_radians = float('inf')
        self.x_threshold = 5.0
        self.screen_width = 900

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize with the pole hanging down (e.g., pi radians away from upright)
        self.state = np.array([0.0, 0.0, np.pi, 0.0])
        return self.state, {}

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        # For example, reward the agent based on the pole's angle:
        # (cos(theta) is 1 when upright and -1 when hanging down)
        # Unpack state: [cart position, cart velocity, pole angle, pole angular velocity]
        x, x_dot, theta, theta_dot = state
        terminated = bool(x < -self.x_threshold or x > self.x_threshold)
        #reward = -(1*theta**2 + 0.1*theta_dot**2 + 0.001*x**2 + 0.0001*x_dot**2)  # scales reward to [0,1]
        reward = (0.5 + np.cos(theta)*0.5) - 0.002*theta_dot**2 - 100*terminated
        # Optionally, modify termination criteria if needed
        return state, reward, terminated, truncated, info


# Register the new environment
gym.envs.registration.register(
    id='CustomCartPole-v0',
    entry_point="__main__:CustomCartPoleEnv",  #"__main__:CustomCartPoleEnv",
    max_episode_steps=400,
    reward_threshold=475.0,
)
env = gym.make('CustomCartPole-v0')
env = DummyVecEnv([lambda: env])

# Define the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
tts = 50000
model.learn(total_timesteps=tts)

# Save the model
model.save(f"ppo_agents/ppo_cartpole_hybrid_ts{tts//1000}k")
'''