# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from torch import Tensor

class LMHead(nn.Linear):
    # This class merely exists to do param initialization correctly.
    # I will probably replace this with a better implementation down the line
    # but for now this can suffice ;)
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)