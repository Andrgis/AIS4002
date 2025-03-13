import numpy as np
import random


class GridworldEnvironment:
    def __init__(self, grid_map):
        """
        grid_map: list of strings, each string is a row with space-separated tokens.
                  Symbols:
                    '#' : wall (impassable)
                    '.' : open space (step penalty)
                    'X' : lava (high negative reward)
                    'O' : exit (goal, positive reward)
        """
        self.raw_map = grid_map
        # parse the map into a list of lists
        self.grid = [row.split() for row in grid_map]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        # Define rewards for each type of cell
        self.rewards = {
            '.': -1,
            '#': -1,
            'X': -10,
            'O': 10
        }
        # Allowed actions and their movement deltas.
        # Here, actions: 0: North, 1: East, 2: South, 3: West
        self.actions = {
            0: (0, -1),  # North: decrease y
            1: (1, 0),   # East: increase x
            2: (0, 1),   # South: increase y
            3: (-1, 0)   # West: decrease x
        }

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, x, y):
        return self.grid[y][x] == '#'

    def get_state_index(self, x, y):
        # Only consider states inside the inner grid (non-wall boundary)
        # You can change this if your entire grid is used as states.
        return y * self.width + x

    def get_cell(self, x, y):
        return self.grid[y][x]

    def get_reward(self, x, y):
        cell = self.get_cell(x, y)
        return self.rewards[cell]

    def next_state(self, x, y, action):
        """
        Given current coordinates (x,y) and an action index,
        returns the next state (nx, ny) after attempting to move.
        If the move hits a wall or is out-of-bounds, the agent stays.
        """
        dx, dy = self.actions[action]
        nx, ny = x + dx, y + dy
        if self.in_bounds(nx, ny) and not self.is_wall(nx, ny):
            return nx, ny
        # invalid move: remain in place
        return x, y

    def get_all_states(self):
        """
        Return list of (x,y) coordinates that are not walls.
        """
        states = []
        for y in range(self.height):
            for x in range(self.width):
                if not self.is_wall(x, y):
                    states.append((x, y))
        return states


class QLearningAgent:
    def __init__(self, env, alpha=0.5, gamma=0.1, epsilon=0.2):
        """
        env: an instance of GridworldEnvironment
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate for epsilon-greedy policy
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # number of states: using all non-wall cells
        self.states = env.get_all_states()
        self.num_states = self.env.width * self.env.height  # index-based; some indices may never be reached
        self.num_actions = len(env.actions)
        # Initialize Q-table with zeros (using state index as row, action index as column)
        self.Q_table = np.zeros((self.num_states, self.num_actions))

    def state_to_index(self, state):
        x, y = state
        return self.env.get_state_index(x, y)

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        Returns an action index.
        """
        if random.uniform(0, 1) < self.epsilon:
            # explore: choose a random legal action (all actions are legal)
            return random.choice(list(self.env.actions.keys()))
        else:
            # exploit: choose action with highest Q-value (break ties randomly)
            state_idx = self.state_to_index(state)
            q_values = self.Q_table[state_idx]
            max_q = np.max(q_values)
            # get all actions with max_q
            actions_with_max = [a for a, q in enumerate(q_values) if q == max_q]
            return random.choice(actions_with_max)

    def update(self, state, action, next_state, reward):
        """
        Perform the Q-learning update for a given transition.
        """
        s_idx = self.state_to_index(state)
        next_s_idx = self.state_to_index(next_state)
        current_q = self.Q_table[s_idx, action]
        # Q-learning update rule
        best_next_q = np.max(self.Q_table[next_s_idx])
        self.Q_table[s_idx, action] = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)

    def compute_policy(self):
        """
        Return a dictionary mapping state (x,y) to best action (as string)
        """
        policy = {}
        for state in self.states:
            s_idx = self.state_to_index(state)
            q_values = self.Q_table[s_idx]
            best_action = np.argmax(q_values)
            # Map action index to direction letter for clarity
            direction_map = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
            policy[state] = direction_map[best_action]
        return policy

    def train(self, episodes=1000, max_steps=100, start_state=None):
        """
        Train the agent for a given number of episodes.
        Optionally specify a fixed start_state; otherwise, choose randomly among non-wall cells.
        """
        for ep in range(episodes):
            if start_state is None:
                state = random.choice(self.states)
            else:
                state = start_state
            for step in range(max_steps):
                action = self.choose_action(state)
                next_state = self.env.next_state(state[0], state[1], action)
                reward = self.env.get_reward(next_state[0], next_state[1])
                self.update(state, action, next_state, reward)
                state = next_state
                # Optionally: break if the agent reaches a terminal state (e.g., exit 'O')
                if self.env.get_cell(state[0], state[1]) == 'O':
                    break

    def print_Q_table(self):
        # Print Q-values for each non-wall state
        for state in self.states:
            s_idx = self.state_to_index(state)
            print(f"State {state}: {self.Q_table[s_idx]}")


def main():
    # Define your gridworld map (each row is a string with space-separated tokens)
    grid_map = [
        "# # # # # # #",
        "# . . . . O #",
        "# . # # . X #",
        "# . . . . . #",
        "# . X . . X #",
        "# # # # # # #"
    ]
    env = GridworldEnvironment(grid_map)
    agent = QLearningAgent(env, alpha=0.5, gamma=0.1, epsilon=0.2)

    # Optionally, set a fixed starting state (e.g., (1,4))
    start_state = (1, 4)
    agent.train(episodes=500, max_steps=100, start_state=start_state)

    # Print the learned Q-table
    agent.print_Q_table()

    # Print the derived policy
    policy = agent.compute_policy()
    print("\nLearned Policy (state: action):")
    for state in sorted(policy.keys()):
        print(f"{state}: {policy[state]}")

if __name__ == "__main__":
    main()
