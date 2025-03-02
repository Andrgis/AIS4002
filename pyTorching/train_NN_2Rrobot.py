import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# Define the network architecture.
class InverseKinematicsModel(nn.Module):
    def __init__(self):
        super(InverseKinematicsModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        # Output 4 values: [sin(a1), cos(a1), sin(a2), cos(a2)]
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, dataloader, num_epochs=100, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


if __name__ == '__main__':
    # Load dataset.
    inputs = np.load('inputs.npy')
    outputs = np.load('outputs.npy')

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

    dataset = TensorDataset(inputs_tensor, outputs_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = InverseKinematicsModel()
    train_model(model, dataloader, num_epochs=100, lr=1e-3)

    torch.save(model.state_dict(), 'ik_model.pth')
    print("Model trained and saved as ik_model.pth")
