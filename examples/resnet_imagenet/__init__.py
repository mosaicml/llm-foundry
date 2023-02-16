# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

try:
    from examples.resnet_imagenet.data import (StreamingImageNet,
                                               build_imagenet_dataspec,
                                               check_dataloader)
    from examples.resnet_imagenet.model import build_composer_resnet
except ImportError as e:
    raise ImportError(
        'Please make sure to pip install .[resnet] or .[resnet-cpu] to get the requirements for the ResNet example.'
    ) from e

__all__ = [
    'StreamingImageNet',
    'build_imagenet_dataspec',
    'check_dataloader',
    'build_composer_resnet',
]
