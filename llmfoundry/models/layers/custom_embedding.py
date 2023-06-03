import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class GPTEmbedding(nn.Embedding):
    def forward(self, input: Tensor) -> Tensor:
        print ("input dtype")
        if input.dtype in [torch.LongTensor]:
            return super().forward(input)
        else:
            # We just want a linear interpolation with the embedding
            return F.linear(input, self.weight)