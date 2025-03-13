import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from PPO_example_gpt import CustomCartPoleEnv

# Make sure the network architecture is identical to the one used during training.
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.actor = nn.Sequential(nn.Linear(64, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        features = self.shared(state)
        return self.actor(features), self.critic(features)


# Recreate the agent's model.
state_dim = 4  # For CartPole-v1, state dimension is 4.
action_dim = 2  # For CartPole-v1, action space has 2 discrete actions.
model = ActorCritic(state_dim, action_dim)

# Load saved parameters.
model.load_state_dict(torch.load("ppo_agent_cust_200.pth", map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode.

# Create the environment with rendering enabled.
env = gym.make("CustomCartPole-v0")#, render_mode="human")
state, _ = env.reset()

done = False
r_tot = 0
while not done:
    # Convert state to tensor.
    state_tensor = torch.FloatTensor(state)
    # Get action probabilities.
    probs, _ = model(state_tensor)
    # Sample an action.
    action = torch.distributions.Categorical(probs).sample().item()

    # Step the environment.
    next_state, reward, done, truncated, _ = env.step(action)
    r_tot += reward
    state = next_state
    print(f"Score: {r_tot}")

env.close()

print(f"Simulation finished with a score of {r_tot}")