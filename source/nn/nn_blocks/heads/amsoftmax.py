from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


from .base import BaseHead


class _amsf(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super(_amsf, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels: Optional[torch.Tensor] = None):
        if self.training:
            assert len(x) == len(labels)
            assert torch.min(labels) >= 0
            assert torch.max(labels) < self.out_features
            
            for W in self.fc.parameters():
                W = F.normalize(W, dim=1)

            x = F.normalize(x, dim=1)
            wf = self.fc(x)

            if len(labels.shape) == 1: weighted = torch.diagonal(wf.transpose(0, 1)[labels])
            else: weighted = torch.sum(torch.multiply(labels, wf), dim=1)
            numerator = self.s * (weighted - self.m)

            if len(labels.shape) == 1:
                excl = torch.cat(
                    [torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], 
                    dim=0
                )
            else:
                excl = torch.cat(
                    [torch.cat((wf[i, :torch.argmax(y)], wf[i, torch.argmax(y)+1:])).unsqueeze(0) for i, y in enumerate(labels)], 
                    dim=0
                )

            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)
            return -torch.mean(L)

        else:
            for W in self.fc.parameters():
                W = F.normalize(W, dim=1)
            x = F.normalize(x, dim=1)
            x = self.fc(x)
            return x


class AMSoftmaxHead(BaseHead):
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        emb_size: int = 100, 
        s: float = 30, 
        m: float = 0.4
    ) -> None:
        super().__init__(input_size, output_size, True)
        self.fc = nn.Linear(input_size, emb_size)
        self.amsf = _amsf(emb_size, output_size, s, m)

    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        emb: bool = False
    ) -> torch.Tensor:

        emb_ = self.fc(x)
        if emb: return emb
        out = self.amsf(emb_, labels)
        return out
        