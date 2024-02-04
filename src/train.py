#! /usr/bin/env python3

import torch
import h5py

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss / (batch_idx + 1), 100. * correct / total

if __name__ == "__main__":
    with open("./data/dataset.h5", "rb") as f:
        data = h5py.File(f, "r")
        print(data.keys())
        print(data["data"].shape)
        print(data["labels"].shape)
        print(data["labels"][:10])
        print(data["data"][:10])
        data.close()