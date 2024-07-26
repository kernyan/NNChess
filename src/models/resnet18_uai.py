from typing import NamedTuple

import torch
from torch import nn

class ResidualBlockParams(NamedTuple):
    """Class for ResNet18 block parameters

    :meta private:

    :param in_channels: Number of input channels
    :type in_channels: int
    :param out_channels: Number of output channels
    :type out_channels: int
    :param downsample: Boolean variable to control downsampling
    :type downsample: bool
    :param stride: Stride value
    :type stride: int
    """

    in_channels: int
    out_channels: int
    downsample: bool = False
    stride: int = 1


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, downsample: bool = False, stride: int = 1
    ):
        """Generates a torch float residual block for ResNet18. A block consists of a conv 3x3 ->
        conv 3x3 -> add, where the downsample parameter controls whether to add a conv 1x1
        on the residual path before feeding into the add.

        :param in_channels: Number of input channels
        :type in_channels: int
        :param out_channels: Number of output channels
        :type out_channels: int
        :param downsample: Boolean variable to control downsampling, defaults to False
        :type downsample: bool, optional
        :param stride: Stride value, defaults to 1
        :type stride: int, optional
        """
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        if self.downsample:
            self.conv_down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.add_relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        residual = x if not self.downsample else self.conv_down(x)
        out = self.conv2(self.conv1(x))
        return self.add_relu(residual + out)


class ResidualParams(NamedTuple):
    """Controls the size and number of blocks of a ResNet18 residual.

    :meta private:
    """

    n_blocks: int
    in_channels: int
    out_channels: int
    downsample: bool = True
    stride: int = 1


class Residual(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        in_channels: int,
        out_channels: int,
        downsample: bool = True,
        stride: int = 1,
    ):
        """Initialization function for a torch float Residual

        :param n_blocks: Number of residual blocks
        :type n_blocks: int
        :param in_channels: Number of input channels
        :type in_channels: int
        :param out_channels: Number of output channels
        :type out_channels: int
        :param downsample: Boolean variable to control downsampling, defaults to True
        :type downsample: bool, optional
        :param stride: Stride value, defaults to 1
        :type stride: int, optional
        """
        super().__init__()
        blocks = []
        for block in range(n_blocks):
            residual_block_params = ResidualBlockParams(
                in_channels,
                out_channels,
                downsample if block == 0 else False,
                stride if block == 0 else 1,
            )
            stride = stride if block == 0 else 1
            blocks.append(ResidualBlock(*residual_block_params))
            in_channels = out_channels
        self.residual = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        return self.residual(x)


# Residuals list
residual_params_list = [
    ResidualParams(2, 64, 64, False, 1),  # 56x56
    ResidualParams(2, 64, 128, True, 2),  # 28x28
    ResidualParams(2, 128, 256, True, 2),  # 14x14
    ResidualParams(2, 256, 512, True, 2),  # 7x7
]


class InputBlock(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        assert in_channels == 3 and out_channels == 64, "Why are these parameters?"
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, 64, (7, 7), padding=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
        )

    def forward(self, x: torch.Tensor):
        return self.input(x)


class OutputBlock(nn.Module):
    def __init__(self, in_height: int, in_width: int, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.AvgPool2d((in_height, in_width)),
            nn.Conv2d(in_channels, out_channels, (1, 1), stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class ResNet18(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        residual_params_list: list[ResidualParams] = residual_params_list,
        flatten: bool = True,
        input_block: bool = True,
        flatten_height: int = 7,
        flatten_width: int = 7,
    ):
        """Initialization function for a float torch ResNet50

        :param in_channels: Number of input channels, defaults to 3
        :type in_channels: int, optional
        :param out_channels: Number of output channels, defaults to 1024
        :type out_channels: int, optional
        :param residual_params_list: List of residual parameters, defaults to residual_params_list
        :type residual_params_list: List, optional
        :param flatten: Flag to enable or disable the flatten layer, defaults to True
        :type flatten: bool, optional
        """
        super().__init__()
        self.flatten = flatten
        self.input_block = input_block

        if self.input_block:
            self.input = InputBlock(in_channels=in_channels)
        # residual blocks
        residuals_list = []
        for residual_params in residual_params_list:
            residuals_list.append(Residual(*residual_params))
        self.residuals = nn.Sequential(*residuals_list)
        # output block
        if self.flatten:
            self.output = OutputBlock(
                flatten_height, flatten_width, residual_params_list[-1].out_channels, out_channels
            )
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        if self.input_block:
            x = self.input(x)
        x = self.residuals(x)
        if self.flatten:
            x = self.output(x)
        x = x.squeeze()
        x = self.fc(x)
        return x