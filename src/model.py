import torch
import torch.nn as nn
from torch import Tensor
import transformers

from typing import List

__all__ = ["DNABERT2FC"]


class DNABERT2FC(nn.Module):
    def __init__(
        self, feats_coord: List[int], feats_final: List[int], **kwargs
    ) -> None:
        super(DNABERT2FC, self).__init__()
        # token embedding
        self.dnabert2 = transformers.AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        # coord embedding
        self.fc_coord = FC(feats_coord)
        # final classification/regression
        self.fc_final = FC(feats_final)

    def forward(self, token: Tensor, coord: Tensor) -> Tensor:
        # token embedding by DNABERT2
        hidden_states = self.dnabert2(token)    # ( [B, N, 768], [B, 768] )
        # embedding with mean or max pooling
        # mean_embedding = torch.mean(hidden_states[0], dim=1)      # [B, 768]
        # max_embedding  = torch.max(hidden_states[0], dim=1)[0]    # [B, 768]
        token_embedding = torch.mean(hidden_states[0], dim=1)       # [B, 768]

        # coord embedding
        coord_embedding = self.fc_coord(coord)

        return self.fc_final(token_embedding + coord_embedding)


class FC(nn.Module):
    def __init__(self, feats: List[int], **kwargs) -> None:
        super(FC, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feats[i], feats[i+1]),
                nn.ReLU()
            ) for i in range(len(feats) - 2)
        ])
        self.out = nn.Linear(feats[-2], feats[-1])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers: x = layer(x)
        return self.out(x)
