from typing import Iterable

import torch
import torch.nn as nn
from torch.autograd import Variable


class BaseModel(nn.Module):
    def __init__(
        self,
        encoder_level: nn.Module,
        head_level: nn.Module,
    ) -> None:
        super(BaseModel, self).__init__()
        
        self.encoder = encoder_level
        self.head = head_level

    def forward(self, x: torch.Tensor, emb: bool = False):
        x = self.encoder(x)
        out = self.head(x, emb)
        return out


class PartlyLoadingModel(nn.Module):
    def __init__(self):
        super().__init__()

    def load_state_dict_part(self, state_dict, ignoring = Iterable[str]):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state or name in ignoring:
                continue
            own_state[name].copy_(param.data)
