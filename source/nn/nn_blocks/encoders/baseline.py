import torch
import torch.nn as nn

from .base import BaseEncoder


def Conv1dBlock(in_channels: int, out_channels: int, kernel_size: int):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=1),
        nn.ReLU(True),
        nn.BatchNorm1d(out_channels),
        nn.Dropout1d(0.3),
    )


class StatsPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        std, mean = torch.std_mean(x, dim=-1)
        return torch.hstack([std, mean])


class BioModel(BaseEncoder):
    def __init__(self, input_size: int, num_classes: int, emb_size: int):
        super().__init__()

        self.cnn = nn.Sequential(
            Conv1dBlock(1, 10, 5),
            Conv1dBlock(10, 30, 9),
            Conv1dBlock(30, 30, 15),
            Conv1dBlock(30, 30, 15),
        )
        self.sp = StatsPooling()
        self.emb = nn.Sequential(
            nn.Linear(60, emb_size),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )
        
        self.initialize_weights()

    def forward(self, x: torch.Tensor):
        x = self.cnn(x)
        x = self.sp(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)