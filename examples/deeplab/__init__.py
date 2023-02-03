# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

try:
    from examples.deeplab.data import ADE20k, StreamingADE20k
    from examples.deeplab.model import build_composer_deeplabv3, deeplabv3
    from examples.deeplab.transforms import (PadToSize, PhotometricDistoration,
                                             RandomCropPair, RandomHFlipPair,
                                             RandomResizePair,
                                             build_ade20k_transformations)
except ImportError as e:
    raise ImportError(
        'Please make sure to pip install .[deeplab] or .[deeplab-cpu] to get the requirements for the DeepLab example.'
    ) from e

__all__ = [
    'ADE20k',
    'StreamingADE20k',
    'deeplabv3',
    'build_composer_deeplabv3',
    'build_ade20k_transformations',
    'RandomResizePair',
    'RandomCropPair',
    'RandomHFlipPair',
    'PadToSize',
    'PhotometricDistoration',
]
