import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
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

class ChessBoardEvaluator(nn.Module):
    def __init__(self):
        super(ChessBoardEvaluator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            *[ResidualBlock(64, 64) for _ in range(8)],  # 8 Residual Blocks
            nn.Conv2d(64, 64, kernel_size=1),  # Reduce feature map size without pooling
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8)  # Global average pooling over 8x8 feature maps
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the tensor, excluding the batch dimension
        x = self.fc(x)
        return x

# Instantiate and test the model
model = ChessBoardEvaluator()
sample_input = torch.rand((1, 2, 8, 8))  # Example input with batch size of 1
output = model(sample_input)  # Forward pass
print(f'Output: {output}')
