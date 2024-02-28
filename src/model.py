import torch
import torch.nn as nn
from torch import Tensor
import transformers

from typing import List, Tuple

__all__ = ["DNABERT2FC"]


class DNABERT2FC(nn.Module):
    def __init__(
        self, feats_coord: List[int], feats: List[int], **kwargs
    ) -> None:
        super(DNABERT2FC, self).__init__()
        # DNABERT2, for sequence embedding
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        self.dnabert2 = transformers.AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        # FC, for coord embedding
        self.fc_coord = FC(feats_coord)
        # FC, for final classification/regression
        self.fc = FC(feats)

    def forward(self, sequence: Tuple[str], coord: Tensor) -> Tensor:
        ## sequence embedding by DNABERT2
        # input
        token = self.tokenizer(             # [B, N]
            sequence, return_tensors = 'pt', padding=True
        )["input_ids"]
        # output
        hidden_states = self.dnabert2(token)   # ( [B, N, 768], [B, 768] )
        sequence_output = hidden_states[0]  # [B, N, 768]
        #pooled_output  = hidden_states[1]  # [B, 768]
        # embedding with mean or max pooling
        sequence_embedding = torch.mean(sequence_output, dim=1)     # [B, 768]
        #sequence_embedding = torch.max(sequence_output, dim=1)[0]  # [B, 768]

        ## coord embedding
        coord_embedding = self.fc_coord(coord)

        ## fc
        x = self.fc(sequence_embedding + coord_embedding)

        return x


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
