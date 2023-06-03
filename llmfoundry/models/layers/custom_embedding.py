import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class SharedEmbedding(nn.Embedding):
    def forward(self, input: Tensor, unembed : bool = False) -> Tensor:
        if unembed:
            # We just want a linear interpolation with the embedding's weights
            return F.linear(input, self.weight)
        return super().forward(input)
