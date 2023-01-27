# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""CIFAR image classification dataset.

The CIFAR datasets are a collection of labeled 32x32 colour images. Please refer
to the `CIFAR dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`_ for more
details.
"""

from typing import Any, Callable, Optional

from composer.core import DataSpec
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.utils import dist
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset

__all__ = ['StreamingCIFAR', 'build_cifar10_dataspec']

# Scale by 255 since the collate `pil_image_collate` results in images in range 0-255
# If using ToTensor() and the default collate, remove the scaling by 255
CIFAR10_CHANNEL_MEAN = 0.4914 * 255, 0.4822 * 255, 0.4465 * 255
CIFAR10_CHANNEL_STD = 0.247 * 255, 0.243 * 255, 0.261 * 255


class StreamingCIFAR(StreamingDataset, VisionDataset):
    """CIFAR streaming dataset based on PyTorch's VisionDataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset
            is stored.
        local (str): Local filesystem directory where dataset is cached during
            operation.
        split (str): The dataset split to use, either 'train' or 'test'.
        shuffle (bool): Whether to iterate over the samples in randomized order.
        transform (callable, optional): A function/transform that takes in an
            image and returns a transformed version. Default: ``None``.
        batch_size (int, optional): The batch size that will be used on each
            device's DataLoader. Default: ``None``.
    """

    def __init__(self,
                 remote: str,
                 local: str,
                 split: str,
                 shuffle: bool,
                 transform: Optional[Callable] = None,
                 batch_size: Optional[int] = None) -> None:

        if split not in ['train', 'test']:
            raise ValueError(
                f"split='{split}', but must be one of ['train', 'test'].")

        self.transform = transform

        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         shuffle=shuffle,
                         batch_size=batch_size)

    def __getitem__(self, idx: int) -> Any:
        sample = super().__getitem__(idx)
        image = sample['x']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = sample['y']
        return image, target


def build_cifar10_dataspec(
    data_path: str,
    is_streaming: bool,
    batch_size: int,
    local: Optional[str] = None,
    is_train: bool = True,
    download: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    **dataloader_kwargs: Any,
) -> DataSpec:
    """Builds a CIFAR-10 dataloader with default transforms.

    Args:
        data_path (str): Path to the data; can be a local path or s3 path
        is_streaming (bool): Whether the data is stored locally or remotely
            (e.g. in an S3 bucket).
        batch_size (int): Batch size per device.
        local (str, optional): If using streaming, local filesystem directory
            where data is cached during operation. Default: ``None``.
        is_train (bool): Whether to load the training data or validation data.
            Default: ``True``.
        download (bool, optional): Whether to download the data locally, if
            needed. Default: ``True``.
        drop_last (bool): Drop remainder samples. Default: ``True``.
        shuffle (bool): Shuffle the dataset. Default: ``True``.
        **dataloader_kwargs (Any): Additional settings for the dataloader
            (e.g. num_workers, etc.).
    """
    if is_streaming and not local:
        raise ValueError(
            '`local` argument must be specified if using a streaming dataset.')

    split = 'train' if is_train else 'test'
    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform = None

    if is_streaming:
        assert local is not None  # for type checkings
        dataset = StreamingCIFAR(
            remote=data_path,
            local=local,
            split=split,
            shuffle=shuffle,
            transform=transform,  # type: ignore
            batch_size=batch_size)
        sampler = None
    else:
        with dist.run_local_rank_zero_first():
            dataset = datasets.CIFAR10(root=data_path,
                                       train=is_train,
                                       download=dist.get_local_rank() == 0 and
                                       download,
                                       transform=transform)
        sampler = dist.get_sampler(dataset,
                                   drop_last=drop_last,
                                   shuffle=shuffle)

    device_transform_fn = NormalizationFn(mean=CIFAR10_CHANNEL_MEAN,
                                          std=CIFAR10_CHANNEL_STD)

    return DataSpec(
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=pil_image_collate,
            **dataloader_kwargs,
        ),
        device_transforms=device_transform_fn,
    )
