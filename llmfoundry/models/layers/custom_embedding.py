# Copyright 2022-2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SharedEmbedding(nn.Embedding):

    def forward(self, input: Tensor, unembed: bool = False) -> Tensor:
        if unembed:
            return F.linear(input, self.weight)
        return super().forward(input)
