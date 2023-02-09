# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""ADE20K Semantic segmentation and scene parsing dataset.

Please refer to the `ADE20K dataset <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_ for more details about this
dataset.
"""
import os
import sys
from io import BytesIO
from itertools import islice
from typing import Any, Optional, Tuple

import streaming
import torch
from composer.core import DataSpec
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.utils import dist
from PIL import Image
from streaming import StreamingDataset
from torch.utils.data import DataLoader, Dataset

from examples.deeplab.transforms import (IMAGENET_CHANNEL_MEAN,
                                         IMAGENET_CHANNEL_STD,
                                         build_ade20k_transformations)

__all__ = ['ADE20k', 'StreamingADE20k']


def build_ade20k_dataspec(
    path: str,
    is_streaming: bool = True,
    local: str = '/tmp/mds-cache/mds-ade20k/',
    *,
    batch_size: int,
    split: str = 'train',
    drop_last: bool = True,
    shuffle: bool = True,
    base_size: int = 512,
    min_resize_scale: float = 0.5,
    max_resize_scale: float = 2.0,
    final_size: int = 512,
    ignore_background: bool = True,
    **dataloader_kwargs,
):
    """Builds an ADE20k dataloader.

    Args:
        path (str): Path to S3 bucket if streaming, otherwise path to local
            data directory
        local (str): Local filesystem directory where dataset is cached during
            operation. Default: ``'/tmp/mds-cache/mds-ade20k/'``.
        is_streaming (bool):  If True, use streaming dataset. Default: ``True``.
        batch_size (int): Batch size per device.
        split (str): The dataset split to use either 'train', 'val', or 'test'.
            Default: ``'train```.
        drop_last (bool): Whether to drop last samples. Default: ``True``.
        shuffle (bool): Whether to shuffle the dataset. Default: ``True``.
        base_size (int): Initial size of the image and target before other
            augmentations. Default: ``512``.
        min_resize_scale (float): The minimum value the samples can be rescaled.
            Default: ``0.5``.
        max_resize_scale (float): The maximum value the samples can be rescaled.
            Default: ``2.0``.
        final_size (int): The final size of the image and target.
            Default: ``512``.
        ignore_background (bool): If true, ignore the background class when
            calculating the training loss. Default: ``true``.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the
            dataloader (e.g. num_workers, etc.)
    """
    joint_transform, image_transforms, target_transforms = build_ade20k_transformations(
        split=split,
        base_size=base_size,
        min_resize_scale=min_resize_scale,
        max_resize_scale=max_resize_scale,
        final_size=final_size)

    if is_streaming:
        dataset = streaming.vision.StreamingADE20K(
            remote=path,
            local=local,
            split=split,
            shuffle=shuffle,
            joint_transform=joint_transform,
            transform=image_transforms,
            target_transform=target_transforms,
            batch_size=batch_size)
        sampler = None
    else:
        dataset = ADE20k(datadir=path,
                         split=split,
                         both_transforms=joint_transform,
                         image_transforms=image_transforms,
                         target_transforms=target_transforms)

        sampler = dist.get_sampler(dataset,
                                   drop_last=drop_last,
                                   shuffle=shuffle)
    device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN,
                                          std=IMAGENET_CHANNEL_STD,
                                          ignore_background=ignore_background)

    return DataSpec(
        dataloader=DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              sampler=sampler,
                              drop_last=drop_last,
                              collate_fn=pil_image_collate,
                              **dataloader_kwargs),
        device_transforms=device_transform_fn,
    )


class ADE20k(Dataset):
    """PyTorch Dataset for ADE20k.

    Args:
        datadir (str): the path to the ADE20k folder. Dataset should be in the format <datadir>/ADEChallengeData2016/images
        split (str): The dataset split to use either 'train', 'val', or 'test'. Default: ``'train```.
        both_transforms (torch.nn.Module): transformations to apply to the image and target simultaneously.
            Default: ``None``.
        image_transforms (torch.nn.Module): transformations to apply to the image only. Default: ``None``.
        target_transforms (torch.nn.Module): transformations to apply to the target only. Default ``None``.
    """

    def __init__(self,
                 datadir: str,
                 split: str = 'train',
                 both_transforms: Optional[torch.nn.Module] = None,
                 image_transforms: Optional[torch.nn.Module] = None,
                 target_transforms: Optional[torch.nn.Module] = None):
        super().__init__()
        self.datadir = datadir
        self.both_transforms = both_transforms
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        split_to_dir = {
            'train': 'training',  # map split names to ade20k image directories
            'val': 'validation',
            'test': 'test'
        }
        self.split = split_to_dir[split]

        # Check datadir value
        if self.datadir is None:
            raise ValueError('datadir must be specified')

        # add ADEChallengeData2016 to root dir
        self.datadir = os.path.join(self.datadir, 'ADEChallengeData2016')

        if not os.path.exists(self.datadir):
            raise FileNotFoundError(
                f'datadir path does not exist: {self.datadir}')

        # Check split value
        if self.split not in ['training', 'validation', 'test']:
            raise ValueError(
                f'split must be one of [`training`, `validation`, `test`] but is: {self.split}'
            )

        self.image_dir = os.path.join(self.datadir, 'images', self.split)
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(
                f'ADE20k directory structure is not as expected: {self.image_dir} does not exist'
            )

        self.image_files = os.listdir(self.image_dir)

        # Filter for ADE files
        self.image_files = [f for f in self.image_files if f[:3] == 'ADE']

        # Remove grayscale samples
        if self.split == 'training':
            corrupted_samples = ['00003020', '00001701', '00013508', '00008455']
            for sample in corrupted_samples:
                sample_file = f'ADE_train_{sample}.jpg'
                if sample_file in self.image_files:
                    self.image_files.remove(sample_file)

    def __getitem__(self, index):
        # Load image
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path)

        # Load annotation target if using either train or val splits
        if self.split in ['training', 'validation']:
            target_path = os.path.join(self.datadir, 'annotations', self.split,
                                       image_file.split('.')[0] + '.png')
            target = Image.open(target_path)

            if self.both_transforms:
                image, target = self.both_transforms((image, target))

            if self.target_transforms:
                target = self.target_transforms(target)

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.split in ['training', 'validation']:
            return image, target  # type: ignore
        else:
            return image

    def __len__(self):
        return len(self.image_files)


class StreamingADE20k(StreamingDataset):
    """Implementation of the ADE20k dataset using StreamingDataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        base_size (int): initial size of the image and target before other augmentations. Default: ``512``.
        min_resize_scale (float): the minimum value the samples can be rescaled. Default: ``0.5``.
        max_resize_scale (float): the maximum value the samples can be rescaled. Default: ``2.0``.
        final_size (int): the final size of the image and target. Default: ``512``.
        batch_size (Optional[int]): Hint the batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def decode_uid(self, data: bytes) -> str:
        return data.decode('utf-8')

    def decode_image(self, data: bytes) -> Image.Image:
        return Image.open(BytesIO(data))

    def decode_annotation(self, data: bytes) -> Image.Image:
        return Image.open(BytesIO(data))

    def __init__(self,
                 remote: str,
                 local: str,
                 split: str,
                 shuffle: bool,
                 base_size: int = 512,
                 min_resize_scale: float = 0.5,
                 max_resize_scale: float = 2.0,
                 final_size: int = 512,
                 batch_size: Optional[int] = None):

        # Validation
        if split not in ['train', 'val']:
            raise ValueError(
                f"split='{split}' must be one of ['train', 'val'].")
        if base_size <= 0:
            raise ValueError('base_size must be positive.')
        if min_resize_scale <= 0:
            raise ValueError('min_resize_scale must be positive')
        if max_resize_scale <= 0:
            raise ValueError('max_resize_scale must be positive')
        if max_resize_scale < min_resize_scale:
            raise ValueError(
                'max_resize_scale cannot be less than min_resize_scale')
        if final_size <= 0:
            raise ValueError('final_size must be positive')

        # Build StreamingDataset
        decoders = {
            'image': self.decode_image,
            'annotation': self.decode_annotation,
        }
        super().__init__(remote=os.path.join(remote, split),
                         local=os.path.join(local, split),
                         shuffle=shuffle,
                         decoders=decoders,
                         batch_size=batch_size)

        # Define custom transforms
        self.both_transform, self.image_transform, self.target_transform = build_ade20k_transformations(
            split=split,
            base_size=base_size,
            min_resize_scale=min_resize_scale,
            max_resize_scale=max_resize_scale,
            final_size=final_size)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        obj = super().__getitem__(idx)
        x = obj['image']
        y = obj['annotation']
        if self.both_transform:
            x, y = self.both_transform((x, y))
        if self.image_transform:
            x = self.image_transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


def check_dataloader():
    """Tests if your dataloader is working locally.

    Run `python data.py my_data_path` to test local dataset. Run `python data.py
    s3://my-bucket/my-dir/data /tmp/path/to/local` to test streaming.
    """
    path = sys.argv[1]
    batch_size = 2
    if len(sys.argv) > 2:
        local = sys.argv[2]
        dataspec = build_ade20k_dataspec(path=path,
                                         local=local,
                                         batch_size=batch_size)
    else:
        dataspec = build_ade20k_dataspec(path=path,
                                         is_streaming=False,
                                         batch_size=batch_size)

    print('Running 5 batchs of dataloader')
    for batch_ix, batch in enumerate(islice(dataspec.dataloader, 5)):
        print(
            f'Batch id: {batch_ix}; Image batch shape: {batch[0].shape}; Target batch shape: {batch[1].shape}'
        )


if __name__ == '__main__':
    check_dataloader()
