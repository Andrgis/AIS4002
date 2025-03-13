import math

import torch
import torch.nn as nn
from pyTorching import generateData




# Define the neural network architecture (must match training)
class InverseKinematicsModel(nn.Module):
    def __init__(self):
        super(InverseKinematicsModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # Initialize and load the trained model.
    model = InverseKinematicsModel()
    model.load_state_dict(torch.load('pyTorching/ik_model.pth'))
    model.eval()  # Set model to evaluation mode

    # Prompt user for the desired end-effector coordinates.
    try:
        x = float(input("Enter desired x-coordinate: "))
        y = float(input("Enter desired y-coordinate: "))
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return

    # Prepare the input tensor. Make sure it's shaped as (1, 2)
    input_tensor = torch.tensor([[x, y]], dtype=torch.float32)

    # Run the model to get predicted joint angles.
    with torch.no_grad():
        predicted_angles = model(input_tensor).numpy()[0]

    sa1, ca1, sa2, ca2 = predicted_angles
    a1 = math.atan2(sa1,ca1)
    a2 = math.atan2(sa2, ca2)
    print(f"\nPredicted Joint Angles:")
    print(f"  a1: {a1:.4f} radians")
    print(f"  a2: {a2:.4f} radians")
    xc, yc = generateData.forward_kinematics(a1, a2)
    print(f"\nResulting end-effector position:")
    print(f"  x: {xc:.4f} m")
    print(f"  y: {yc:.4f} m")


if __name__ == '__main__':
    main()

