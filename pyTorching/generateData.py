import math
import random
import numpy as np

# Robot arm link lengths.
l1 = 1.0
l2 = 1.0


def forward_kinematics(a1: float, a2: float) -> tuple:
    """
    Computes forward kinematics for a 2R planar robot.
    """
    x = l1 * math.cos(a1) + l2 * math.cos(a1 + a2)
    y = l1 * math.sin(a1) + l2 * math.sin(a1 + a2)
    return x, y


def inverse_kinematics_elbow_down(x: float, y: float) -> tuple:
    """
    Computes the elbow-down inverse kinematics solution for the given end-effector position.
    """
    # Compute cos(a2) using the law of cosines.
    cos_a2 = (x ** 2 + y ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    # Clamp cos_a2 to avoid numerical issues.
    cos_a2 = max(min(cos_a2, 1.0), -1.0)
    # Choose the elbow-down solution (negative angle).
    a2 = -math.acos(cos_a2)

    # Compute a1 using the two-argument arctan.
    k1 = l1 + l2 * math.cos(a2)
    k2 = l2 * math.sin(a2)
    a1 = math.atan2(y, x) - math.atan2(k2, k1)
    return a1, a2


def generate_dataset(num_samples: int = 10000):
    inputs = []
    outputs = []
    for _ in range(num_samples):
        # Sample a random pair of joint angles.
        a1 = random.uniform(-math.pi, math.pi)
        a2 = random.uniform(-math.pi, math.pi)
        # Compute the end-effector position.
        x, y = forward_kinematics(a1, a2)
        # Compute the inverse kinematics using the elbow-down solution.
        a1_sol, a2_sol = inverse_kinematics_elbow_down(x, y)
        # Represent each angle as (sin, cos)
        outputs.append([math.sin(a1_sol), math.cos(a1_sol),
                        math.sin(a2_sol), math.cos(a2_sol)])
        inputs.append([x, y])

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    return inputs, outputs


if __name__ == '__main__':
    inputs, outputs = generate_dataset(num_samples=10000)
    np.save('inputs.npy', inputs)
    np.save('outputs.npy', outputs)
    print("Data generated and saved as inputs.npy and outputs.npy")
