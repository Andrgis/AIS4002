import numpy as np
import random

# directions
'''dir = ("N","E","S","W")
x_dir = (0,1,0,-1)
y_dir = (-1,0,1,0)'''
dir = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0)
}

x, y = 1, 4

# defining environment:
_map = ["# # # # # # #",
        "# . . . . O #",
        "# . # # . X #",
        "# . . . . . #",
        "# . X . . X #",
        "# # # # # # #"]
world = [x.split() for x in _map]
w, h = len(_map[0]), len(_map)
worldA = world

# Constants
alpha = 0.5  # Learning rate
omega = 0.1  # Discount factor

# Initializing Q-table (h*w)xa:
n_states = (h-2)*(w-2)
n_actions = len(dir)
np.zeros((n_states, n_actions))


# Coordinates mapped to State definition: s = w*y+x
def state(coords: tuple[int, int]) -> tuple[int, list]:
    si = w * coords[1] + coords[0]
    world_state = [x.split() for x in _map]
    world_state[coords[1]][coords[0]] = 'A'
    return si, world_state


# Actions: Move: N, E, S, W
def move(direction: str):
    dx, dy = dir[direction]
    global x, y
    nx, ny = x + dx, y + dy
    cell_value = world[ny][nx]
    if cell_value != '#':
        x = nx
        y = ny
        si, worldA = state((x, y))


def reward(cell_value: str) -> int:
    r: int = 0
    if cell_value == ".":
        r = -1
    elif cell_value == "#":
        r = -1
    elif cell_value == "X":
        r = -10
    elif cell_value == "O":
        r = 10
    return r

def Q_learning(Q_table, epochs: int = 100):
    for epoch in epochs:
        si = np.random.randint(0, n_states)
#
def main() -> None:
    si, worldA_test = state((3, 3))
    [print(*x) for x in worldA_test]


if __name__ == "__main__":
    main()
