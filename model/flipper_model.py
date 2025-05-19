import torch
import torch.nn as nn

class FlipperModel(nn.Module):
    def __init__(self, input_neurons, output_neurons):
        super(FlipperModel, self).__init__()

        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.fc1 = nn.Linear(input_neurons, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, output_neurons)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))

        return x