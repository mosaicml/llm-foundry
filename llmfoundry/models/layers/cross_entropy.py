# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

try:
    from flash_attn.losses.cross_entropy import SoftmaxCrossEntropyLossFn
except:
    SoftmaxCrossEntropyLossFn = None

import torch.nn as nn


class FusedCrossEntropyLoss(nn.Module):

    def __init__(
        self,
        ignore_index=-100,
        reduction='mean',
        label_smoothing=0.0,
        inplace_backward=False,
        process_group=None,
    ):
        if SoftmaxCrossEntropyLossFn is None:
            raise ValueError(
                'Fused Cross Entropy is not installed. Either (1) have a CUDA-compatible GPU '
                +
                'and `pip install .[gpu]` if installing from source or `pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v1.0.3#subdirectory=csrc/xentropy` '
                +
                'if installing from pypi, or (2) set your config model.loss_fn=torch_crossentropy.'
            )
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.inplace_backward = inplace_backward
        self.process_group = process_group

    def forward(self, input, target):
        assert input.is_cuda and target.is_cuda
        # SoftmaxCrossEntropyLoss implicitly casts to float
        loss = SoftmaxCrossEntropyLossFn.apply(
            input,
            target,
            self.label_smoothing,
            self.ignore_index,
            self.inplace_backward,
            self.process_group,
        )
        if self.reduction == 'mean':
            return loss.sum() / (target != self.ignore_index).sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
