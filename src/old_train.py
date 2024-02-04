#! /usr/bin/env python3

import torch
from torch import nn, optim
from torch.utils.data import Dataset, random_split, Subset, DataLoader
import h5py
import numpy as np
import pandas as pd


def save_to_csv(train_losses, val_losses, filename):
    df = pd.DataFrame({"train_losses": train_losses, "val_losses": val_losses})
    df.to_csv(filename, index=False)


class hdf5Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = None
        self.labels = None
        with h5py.File(self.path, "r") as f:
            self.len = len(f["data"])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.data is None or self.labels is None:
            with h5py.File(self.path, "r") as f:
                data = torch.from_numpy(f["data"][idx]).float()
                label = torch.unsqueeze(
                    torch.tensor(f["labels"][idx], dtype=torch.float), 0
                )
        return data, label


class Model(nn.Module):
    def __init__(self):
        self.INPUT_SIZE = 68
        super(Model, self).__init__()
        self.fc1 = nn.Linear(self.INPUT_SIZE, 4096)
        self.norm1 = nn.LayerNorm(4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.norm2 = nn.LayerNorm(2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.norm3 = nn.LayerNorm(1024)
        self.fc4 = nn.Linear(1024, 512)
        self.norm4 = nn.LayerNorm(512)
        self.fc5 = nn.Linear(512, 256)
        self.norm5 = nn.LayerNorm(256)
        self.fc6 = nn.Linear(256, 128)
        self.norm6 = nn.LayerNorm(128)
        self.fc7 = nn.Linear(128, 64)
        self.norm7 = nn.LayerNorm(64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.norm3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.norm4(self.fc4(x)))
        x = self.dropout(x)
        x = self.relu(self.norm5(self.fc5(x)))
        x = self.dropout(x)
        x = self.relu(self.norm6(self.fc6(x)))
        x = self.dropout(x)
        x = self.relu(self.norm7(self.fc7(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x


class ModelSmall(nn.Module):
    def __init__(self):
        self.INPUT_SIZE = 68
        super(ModelSmall, self).__init__()
        self.fc1 = nn.Linear(self.INPUT_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = torch.tanh(self.output(x))
        return x


def train_and_validate(model, train_dataset, val_dataset, criterion, optimizer, device):
    model.to(device)
    best_val_lost = float("inf")

    train_losses = []
    val_losses = []

    epochs = 10

    for epoch in range(epochs):
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

            if batch_idx % (10) == 0:
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
                    torch.save(model.state_dict(), f"./data/model_{epoch}_{MODEL}.pth")
                    print(
                        f"Model saved at ./data/model_{epoch}_{MODEL}.pth, epoch {epoch} with loss: {avg_val_loss:.6f}"
                    )
                    save_to_csv(train_losses, val_losses, f"./data/losses_{MODEL}.csv")

    torch.save(model.state_dict(), f"./data/model_final_{MODEL}.pth")
    print(f"Model saved at ./data/model_final_{MODEL}.pth")


BATCH_SIZE = 4096
CHOICE = "model"
model_list = {
    "p_linear_1mil": Model,
    "p_linear_200k": ModelSmall,
}

MODEL_FILE = "dataset_80000_games"
MODEL = f"dataset_80000_games_{CHOICE}"

if __name__ == "__main__":
    loader = hdf5Dataset(f"./data/{MODEL_FILE}.h5")
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

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_and_validate(
        model, train_dataloader, val_subset, criterion, optimizer, "cuda"
    )
