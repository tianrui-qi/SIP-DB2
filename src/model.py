import torch
import torch.nn as nn
from torch import Tensor

from typing import List

__all__ = ["FCN"]


class FCN(nn.Module):
    def __init__(self, feats: List[int]) -> None:
        super(FCN, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(feats[i], feats[i+1]) for i in range(len(feats) - 1)]
        )
        # output
        self.fc = nn.Linear(self.feats[-1], 1)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers: x = layer(x)
        x = self.fc(x)
        return x
