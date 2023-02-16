# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

try:
    from examples.resnet_cifar.data import (StreamingCIFAR,
                                            build_cifar10_dataspec)
    from examples.resnet_cifar.model import (ResNetCIFAR,
                                             build_composer_resnet_cifar)
except ImportError as e:
    raise ImportError(
        'Please make sure to pip install .[cifar] or .[cifar-cpu] to get the requirements for the CIFAR example.'
    ) from e

__all__ = [
    'StreamingCIFAR',
    'build_cifar10_dataspec',
    'ResNetCIFAR',
    'build_composer_resnet_cifar',
]
