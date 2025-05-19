import os

import torch
import torch.nn as nn
import numpy as np
import config.config as config

import matplotlib.pyplot as plt

from builder.flipper_builder import build_flipper_dataset
from model.flipper_model import FlipperModel

from provider.flipper_provider import get_loader

from tqdm import tqdm



def train(loader, loss_fn, optimizer, model):
    torch.enable_grad()
    model.train()

    losses = []

    loop = tqdm(loader)

    for (data, target) in loop:
        optimizer.zero_grad(set_to_none=True)
        data = model(data)

        loss = loss_fn(data, target)

        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        losses.append(loss_value)

    return np.mean(losses)


def validate(loader, loss_fn, model):
    model.eval()

    losses = []

    loop = tqdm(loader)

    for (data, target) in loop:
        data = model(data)

        loss = loss_fn(data, target)
        loss_value = loss.item()

        losses.append(loss_value)

    return np.mean(losses)


def run():
    model = FlipperModel(input_neurons=10, output_neurons=10)

    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    training_loader = get_loader(config.get_training_path())
    validation_loader = get_loader(config.get_validation_path())

    training_losses = []
    validation_losses = []

    for epoch in range(0, 1000):
        training_loss = train(training_loader, loss_function, optimizer, model)
        training_losses.append(training_loss)

        validation_loss = validate(validation_loader, loss_function, model)
        validation_losses.append(validation_loss)

    plt.plot(training_losses, label="training")
    plt.plot(validation_losses, label="validation")
    plt.show()

    array = torch.Tensor([0, 1, 0, 1, 1, 1, 1, 0, 0, 0])
    print(torch.round(model(array)))


if __name__ == '__main__':
    run()



