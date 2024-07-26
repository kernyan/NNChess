import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Resnet883Board(nn.Module):
    def __init__(self):
        super(Resnet883Board, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),  # 3 input channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(128, 128) for _ in range(10)]
        )
        self.transition_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.final_pooling = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Change from (batch, height, width, channels) to (batch, channels, height, width)
        x = self.initial_conv(x)
        x = self.residual_blocks(x)
        x = self.transition_conv(x)
        x = self.final_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
