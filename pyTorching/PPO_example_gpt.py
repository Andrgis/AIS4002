import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# PPO Hyperparameters
GAMMA = 0.99  # Discount factor
LR = 3e-4  # Learning rate
EPSILON = 0.2  # Clipping parameter for PPO
EPOCHS = 8  # Number of training epochs per update
BATCH_SIZE = 64  # Minibatch size for updates
UPDATE_INTERVAL = 2000  # Steps before updating policy
TAU = 0.95  # GAE smoothing factor


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
        reward = (0.5 + np.cos(theta)*0.5) - 0.003*theta_dot**2 - 100*terminated
        # Optionally, modify termination criteria if needed
        return state, reward, terminated, truncated, info


# Register the new environment
gym.envs.registration.register(
    id='CustomCartPole-v0',
    entry_point="PPO_example_gpt:CustomCartPoleEnv",  #"__main__:CustomCartPoleEnv",
    max_episode_steps=400,
    reward_threshold=475.0,
)

# Use the customized environment
env = gym.make('CustomCartPole-v0')


# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.actor = nn.Sequential(nn.Linear(64, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Linear(64, 1)  # Value function output

    def forward(self, state):
        features = self.shared(state)
        return self.actor(features), self.critic(features)


# PPO Agent
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        self.memory = []  # Stores trajectory data
        self.reward_memory = []

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        probs, _ = self.model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def store_transition(self, transition):
        self.memory.append(transition)

    def compute_advantages(self, rewards, values, masks):
        advantages, returns = [], []
        gae = 0
        next_value = 0  # Bootstrap next value for GAE

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + GAMMA * next_value * masks[step] - values[step]
            gae = delta + GAMMA * TAU * masks[step] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        return advantages, returns

    def update_policy(self):
        # Convert memory to tensors
        states, actions, log_probs, rewards, values, masks = zip(*self.memory)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.stack(log_probs).detach()
        values = torch.FloatTensor(values).to(self.device)

        # Compute advantages
        advantages, returns = self.compute_advantages(rewards, values, masks)

        for _ in range(EPOCHS):
            for i in range(0, len(states), BATCH_SIZE):
                idx = slice(i, i + BATCH_SIZE)

                # Get new policy probabilities
                probs, new_values = self.model(states[idx])
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions[idx])

                # Compute policy ratio
                ratio = (new_log_probs - old_log_probs[idx]).exp()

                # Compute clipped objective
                unclipped = ratio * advantages[idx]
                clipped = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value loss
                value_loss = (returns[idx] - new_values.squeeze()).pow(2).mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory = []  # Clear memory

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            for t in range(1, UPDATE_INTERVAL + 1):
                action, log_prob = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                mask = 1.0 - float(done or truncated)  # Mask for terminal state

                _, value = self.model(torch.FloatTensor(state).to(self.device))
                self.store_transition((state, action, log_prob, reward, value.item(), mask))

                state = next_state
                episode_reward += reward

                if done or truncated:
                    break

            # Update the policy every UPDATE_INTERVAL steps
            self.reward_memory.append(episode_reward)
            self.update_policy()
            print(f"Episode {episode + 1}: Reward = {episode_reward}")


# Run PPO on CartPole-v1
#env = gym.make("CartPole-v1")
agent = PPOAgent(env)
n_train = 8
agent.train(n_train)

plt.plot(range(len(agent.reward_memory)), agent.reward_memory)
plt.grid()
plt.xlabel("# Episodes")
plt.ylabel("Reward")
plt.show()


# After training is complete

torch.save(agent.model.state_dict(), f"ppo_agents/ppo_agent_swingup_{n_train}.pth")
print("Agent saved!")
