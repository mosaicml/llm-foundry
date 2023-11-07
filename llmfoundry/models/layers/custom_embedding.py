# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SharedEmbedding(nn.Embedding):
    # This class enables weight tying of the input and output embedding
    # layers, following standard practice https://paperswithcode.com/method/weight-tying
    def forward(self, input: Tensor, unembed: bool = False) -> Tensor:
        
        # if unembed, simply pass through a linear layer
        if unembed:
            return F.linear(input, self.weight)
        return super().forward(input)
