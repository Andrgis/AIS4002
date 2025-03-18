import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv  # base class for convenience
import numpy as np
import math
from gymnasium import spaces

# Physical parameters (from your simulation)
m_p = 0.1       # Pole mass [kg]
m_c = 0.01       # Cart mass [kg] (you can adjust)
l = 0.095       # Pole length [m]
l_com = l / 2   # Distance to center of mass of pole
J = (1/3) * m_p * l * l  # Inertia of the pole [kg*m^2]
g = 9.81        # Gravitational acceleration [m/s^2]
total_mass = m_p + m_c
polemass_length = m_p * l_com

# Swingup parameters
Er = 0.015      # Reference energy (J)
ke = 5         # Gain for swingup
u_max = 2.5     # Maximum acceleration [m/s^2]
balance_range = 180.0  # Degrees: within this range, use balance mode

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

        self.iu = 0
        self.top_reached = False

        # Define a continuous action space (e.g., acceleration in m/s^2)
        self.action_space = spaces.Discrete(9)
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

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
        # Map discrete actions to acceleration values.
        acceleration_map = {
            0: -2.5,
            1: -1.25,
            2: 0.0,
            3: 1.25,
            4: 2.5,
            5: 3.75,
            6: 5.0,
            7: -3.75,
            8: -5.0
        }
        applied_acc = acceleration_map[action]

        # Unpack current state
        x, x_dot, theta, theta_dot = self.state

        # Convert pole angle to degrees for mode switching
        theta_deg = math.degrees((theta + math.pi) % (2*math.pi) - math.pi)

        if not self.top_reached:
            self.top_reached = abs(theta_deg) < 5.0

        # Determine control mode:
        if abs(theta_deg) > balance_range:
            # Swing-up mode: use physics-based swingup to get acceleration
            u = swingup_acceleration(theta)
            #print(f"Swing-up: {u}")
        else:
            # Balance mode: agent provides acceleration (action)
            # Here we assume the agent's action is a scalar in [-1,1] and scale it to the physical limits.
            # You may choose to tune this scaling.
            #u = np.clip(action, -1.0, 1.0) * u_max
            u = acceleration_map[action]
            self.iu += 1
            #print(f"Agent control: {u}")


        # Now, compute dynamics.
        # Instead of calling super().step, we integrate our own equations.
        # Using the equations similar to CartPole but replacing force with m_c * u.
        force = m_c * u  # force = mass_cart * acceleration

        # Using the dynamics from the original CartPole:
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
        thetaacc = (g * sintheta - costheta * temp) / (l * (4.0/3.0 - m_p * costheta**2 / total_mass)) - 0.5*theta_dot
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Update state using Euler integration
        x = x + dt * x_dot
        x_dot = x_dot + dt * xacc
        theta = theta + dt * theta_dot
        theta_dot = theta_dot + dt * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

        # Termination: Only if cart goes out of bounds
        done = bool(x < -self.x_threshold or x > self.x_threshold)
        '''if abs(theta_deg) > 90 and self.iu > 10:
            done = True'''
        # Define reward: you could reward based on how close the pole is to upright.
        # For example, reward = 1 when pole is perfectly upright (theta ~ 0).

        if self.top_reached:
            reward = np.cos(theta) - 0.002 * theta_dot**2
        else:
            reward = (0.5 + 0.5 * np.cos(theta)) - 0.002 * theta_dot**2

        '''reward = np.clip(2-2*abs(theta_deg)/balance_range, 0, 1)
        if abs(theta_deg) < balance_range:
            reward -= 0.01 * abs(theta_dot)
        #else:
        #    reward -= 0.3'''
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
    max_episode_steps=200,
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
        print(reward)
        env.render()
    env.close()
    print("Episode finished with total reward:", total_reward)