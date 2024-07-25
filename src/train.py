#! /usr/bin/env python3

import torch
from torch import nn, optim
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from torch.nn import functional as F
import numpy as np
import pandas as pd


def save_to_csv(train_losses, val_losses, filename):
    df = pd.DataFrame({"train_losses": train_losses, "val_losses": val_losses})
    df.to_csv(filename, index=False)


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
        return torch.from_numpy(self.data[idx]).float(), torch.unsqueeze(
            torch.tensor(self.label[idx]).float(), dim=0
        )


class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.a1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=1)
        self.d2 = nn.Conv2d(128, 128, kernel_size=1)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1)

        self.last = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))

        # 4x4
        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))

        # 2x2
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        # 1x128
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        x = x.view(-1, 128)
        x = self.last(x)

        # value output
        return F.tanh(x)


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
                    torch.save(model.state_dict(), f"./data/model_{MODEL}.pth")
                    print(
                        f"Model saved at ./data/model_{MODEL}.pth, epoch {epoch} with loss: {avg_val_loss:.6f}"
                    )
                    save_to_csv(train_losses, val_losses, f"./data/losses_{MODEL}.csv")

    torch.save(model.state_dict(), f"./data/model_final_{MODEL}.pth")
    print(f"Model saved at ./data/model_final_{MODEL}.pth")


BATCH_SIZE = 256
CHOICE = "p_conv2d_small"
model_list = {
    "p_conv2d_small": ConvModel,
}

MODEL_FILE = "./data/dataset_5mil_games.npz"
MODEL = f"dataset_56000_{CHOICE}"

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
    print(
        f"Model {CHOICE} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters"
    )

    optimizer = optim.Adam(model.parameters())
    train_and_validate(
        model, train_dataloader, val_subset, criterion, optimizer, "cuda"
    )
