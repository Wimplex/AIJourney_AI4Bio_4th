# https://github.com/csho33/bacteria-ID

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEncoder


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5,
            stride=stride, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
            stride=1, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def stats_pool(x: torch.Tensor) -> torch.Tensor:
    std, mean = torch.std_mean(x, dim=2)
    return torch.concat([std, mean], dim=1)


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


class ResNet1D(BaseEncoder):
    def __init__(
        self, 
        hidden_sizes: int, 
        num_blocks: int, 
        input_dim: int = 1000,
        in_channels: int = 1, 
        n_classes: int = 30, 
    ):
        super().__init__(input_dim)
        assert len(num_blocks) == len(hidden_sizes)
        self.input_dim = input_dim
        self.__in_channels = 64
        self.n_classes = n_classes
        
        self.conv1 = nn.Conv1d(in_channels, self.__in_channels, kernel_size=5, stride=1,
            padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Flexible number of residual encoding layers
        layers = []
        strides = [1] + [2] * (len(hidden_sizes) - 1)
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(self._make_layer(hidden_size, num_blocks[idx],
                stride=strides[idx]))
        self.encoder = nn.Sequential(*layers)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.encoder(x)
        z = flatten(x)
        # z = stats_pool(x)
        return z

    def forward(self, x):
        z = self.encode(x)
        return z

    def _make_layer(self, out_channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.__in_channels, out_channels,
                stride=stride))
            self.__in_channels = out_channels
        return nn.Sequential(*blocks)