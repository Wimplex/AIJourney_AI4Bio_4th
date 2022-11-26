from typing import Iterable

import torch
import torch.nn as nn

from .base import BaseHead


class LinearHead(BaseHead):
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        sizes: Iterable[int] = None
    ) -> None:
        super(LinearHead, self).__init__(input_size, output_size, False)

        if sizes:
            blocks = []
            sizes = [input_size] + sizes + [output_size]
            for i in range(len(sizes) - 1):
                layer = nn.Linear(sizes[i], sizes[i + 1])
                blocks.append(layer)
            self.fc = nn.ModuleList(blocks)

        else:
            self.fc = nn.ModuleList([nn.Linear(input_size, output_size)])

    def forward(self, x: torch.Tensor, emb: bool = False) -> torch.Tensor:
        for i, layer in enumerate(self.fc):
            if len(self.fc) > 1 and emb and i == len(self.fc) - 1:
                break
            x = layer(x)
        return x