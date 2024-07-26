#! /usr/bin/env python3

from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from torch.nn import functional as F
import numpy as np
import pandas as pd

from models.resnet import ResnetBoard
from models.resnet_883 import Resnet883Board
from models.convnet import Conv2dBoard
from models.resnet18_uai import ResNet18


def save_to_csv(train_losses, val_losses, filename):
    df = pd.DataFrame({"train_losses": train_losses, "val_losses": val_losses})
    df.to_csv(filename, index=False)


def pad_to_224(tensor):
    pad_h = 224 - 8
    pad_w = 224 - 8
    if len(tensor.shape) == 1:
        return tensor
    padded_tensor = F.pad(tensor, (0, pad_h, 0, pad_w), mode="constant", value=0)
    return padded_tensor

class ChessDataset(Dataset):
    def __init__(self, path):
        o = np.load(path)
        self.data = o["data"]
        self.label = o["labels"]
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # convert to float
        o = torch.from_numpy(self.data[idx]).float(), torch.unsqueeze(
            torch.tensor(self.label[idx]).float(), dim=0
        )
        return [pad_to_224(x) for x in o]


def train_and_validate(model, train_dataset, val_dataset, criterion, optimizer, device):
    model.to(device)
    best_val_lost = float("inf")

    train_losses = []
    val_losses = []

    EPOCHS = 100

    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_total = 0

        total_val_loss = 0
        val_total = 0

        print(len(train_dataset))
        for batch_idx, (inputs, targets) in enumerate(train_dataset):
            model.train()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_total += inputs.size(0)

            if batch_idx % 1000 == 0:
                avg_loss = train_loss / train_total
                train_losses.append(avg_loss)
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Train Loss: {avg_loss:.6f}")

                model.eval()
                with torch.no_grad():
                    for inputs, targets in val_dataset:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        total_val_loss += loss.item() * inputs.size(0)
                        val_total += inputs.size(0)

                avg_val_loss = total_val_loss / val_total
                val_losses.append(avg_val_loss)
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Val Loss: {avg_val_loss:.6f}"
                )

                if avg_val_loss < best_val_lost:
                    best_val_lost = avg_val_loss
                    torch.save(model.state_dict(), MODEL_WEIGHT)
                    print(
                        f"Model saved at ./data/model_{MODEL}.pth, epoch {epoch} with loss: {avg_val_loss:.6f}"
                    )
                    save_to_csv(train_losses, val_losses, f"./data/losses_{MODEL}.csv")

    torch.save(model.state_dict(), f"./data/model_final_{MODEL}.pth")
    print(f"Model saved at ./data/model_final_{MODEL}.pth")


BATCH_SIZE = 256
CHOICE = "resnet_uai"
model_list = {
    "convnet": Conv2dBoard,
    "resnet": ResnetBoard,
    "resnet_883": Resnet883Board,
    "resnet_uai": ResNet18,
}

#MODEL_FILE = "./data/dataset_5mil_games.npz"
MODEL_FILE = "./data/dataset_5mil_games_388.npz"
MODEL = f"dataset_56000_{CHOICE}"
MODEL_WEIGHT = f"./data/model_{MODEL}.pth"

if __name__ == "__main__":
    loader = ChessDataset(MODEL_FILE)
    total_size = len(loader)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(loader, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_indices = np.random.choice(val_size, int(val_size * 0.01), replace=False)
    val_subset = DataLoader(
        Subset(val_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=True
    )

    criterion = nn.MSELoss()
    model = model_list[CHOICE]()

    if Path(MODEL_WEIGHT).exists():
        checkpoint = torch.load(MODEL_WEIGHT)
        model.load_state_dict(checkpoint)

    print(
        f"Model {CHOICE} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters"
    )

    optimizer = optim.Adam(model.parameters())
    train_and_validate(
        model, train_dataloader, val_subset, criterion, optimizer, "cuda"
    )
