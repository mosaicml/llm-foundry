# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""The CIFAR ResNet torch module.

See the :doc:`Model Card </model_cards/resnet>` for more details.
"""

# Code below adapted from https://github.com/facebookresearch/open_lth
# and https://github.com/pytorch/vision

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models.tasks import ComposerClassifier

__all__ = ['ResNetCIFAR', 'build_composer_resnet_cifar']


class ResNetCIFAR(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample: bool = False):
            super(ResNetCIFAR.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(f_in,
                                   f_out,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = nn.Conv2d(f_out,
                                   f_out,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
            self.bn2 = nn.BatchNorm2d(f_out)
            self.relu = nn.ReLU(inplace=True)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out),
                )
            else:
                self.shortcut = nn.Identity()

        def forward(self, x: torch.Tensor):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return self.relu(out)

    def __init__(self,
                 plan: List[Tuple[int, int]],
                 initializer: Optional[Callable],
                 outputs: int = 10):
        super(ResNetCIFAR, self).__init__()
        outputs = outputs or 10

        self.num_classes = outputs

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3,
                              current_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(current_filters)
        self.relu = nn.ReLU(inplace=True)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(
                    ResNetCIFAR.Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], outputs)

        if initializer is not None:
            self.apply(initializer)

    def forward(self, x: torch.Tensor):
        out = self.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @staticmethod
    def is_valid_model_name(model_name: str):
        return model_name in [f'resnet_{layers}' for layers in (20, 56)]

    @staticmethod
    def get_model_from_name(model_name: str,
                            initializer: Optional[Callable] = None,
                            num_classes: int = 10):
        """The naming scheme for a ResNet is ``'resnet_D'``.

        D is the model depth (e.g. ``'resnet_56'``)
        """
        if not ResNetCIFAR.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        depth = int(
            model_name.split('_')[-1])  # for resnet56, depth 56, width 16
        width = 16

        if (depth - 2) % 3 != 0:
            raise ValueError('Invalid ResNetCIFAR depth: {}'.format(depth))
        num_blocks = (depth - 2) // 6

        model_arch = {
            56: [(width, num_blocks), (2 * width, num_blocks),
                 (4 * width, num_blocks)],
            20: [(width, num_blocks), (2 * width, num_blocks),
                 (4 * width, num_blocks)],
        }

        return ResNetCIFAR(model_arch[depth], initializer, num_classes)


def build_composer_resnet_cifar(model_name: str,
                                num_classes=10) -> ComposerClassifier:
    """Factory function to produce a CIFAR ResNet ComposerModel.

    Args:
        model_name (str): one of ['resnet_20', 'resnet_56']
        num_classes (int): number of classes to use for the output softmax
    """

    def weight_init(w: torch.nn.Module):
        if isinstance(w, nn.Linear) or isinstance(w, nn.Conv2d):
            torch.nn.init.kaiming_normal_(w.weight)
        if isinstance(w, nn.BatchNorm2d):
            w.weight.data = torch.rand(w.weight.data.shape)
            w.bias.data = torch.zeros_like(w.bias.data)

    model = ResNetCIFAR.get_model_from_name(model_name=model_name,
                                            initializer=weight_init,
                                            num_classes=num_classes)
    composer_model = ComposerClassifier(module=model)
    return composer_model
