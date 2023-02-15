# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import torch
from composer.utils import dist
from torch.utils.data import DataLoader, Dataset


class SyntheticImageCaptionDataset(Dataset):
    """Synthetic Image Captioning dataset.

    Imitates a dataset of images plus captions.

    Args:
        image_size (int): Size of the images. Default: `512`.
        num_samples (int): Number of samples in the dataset. Default: `10000`.
    """

    def __init__(self, image_size: int = 512, num_samples: int = 10000):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Generate image input
        image = torch.randn(3, self.image_size, self.image_size)

        # Generate caption input
        caption = torch.randint(0, 128, (77,), dtype=torch.long)

        sample = {'image_tensor': image, 'input_ids': caption}
        return sample


def build_synthetic_dataloader(image_size: int = 512,
                               num_samples: int = 10000,
                               batch_size: int = 32,
                               drop_last: bool = True,
                               shuffle: bool = True,
                               **dataloader_kwargs):
    dataset = SyntheticImageCaptionDataset(image_size=image_size,
                                           num_samples=num_samples)
    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader
